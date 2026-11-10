#!/bin/bash
# Create, check and delete the GKE cluster for the week 2 GPU session.
#
#   bash scripts/gpu/cluster.sh image   # build app:v3 with Cloud Build (no GPU, do it the day before)
#   bash scripts/gpu/cluster.sh grab    # reserve one L4 in the first zone that has one (retries until it does)
#   bash scripts/gpu/cluster.sh up      # cluster + L4 node pool (uses the reservation) + model disk + app pod
#   bash scripts/gpu/cluster.sh down    # delete everything, including the reservation, then run check
#   bash scripts/gpu/cluster.sh check   # list anything still billable; all counts should be 0
#
# Why a reservation: when L4s are scarce, a plain request grabs a GPU in about a minute,
# but GKE's node pool request loses the race and then retries silently for 30+ minutes.
# A reservation holds the GPU for this project from the moment it succeeds, and the node
# pool consumes exactly that reservation, so there is no race. It bills like a running VM
# (~$0.85/h) from creation until deleted, so `down` deletes it first.
set -euo pipefail
PROJECT=${PROJECT:-docsync-vllm-anagha}
ZONE=${ZONE:-$(cat /tmp/l4_zone 2>/dev/null || echo us-central1-a)}
REGION=us-central1
CLUSTER=docsync
RES=docsync-l4
REPO=$REGION-docker.pkg.dev/$PROJECT/docsync
ZONES=${ZONES:-"us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west4-a us-west4-c"}

gcloud config set project "$PROJECT" >/dev/null

reserve() {   # reserve ZONE: try without, then with, an explicit accelerator (G2 attaches its L4 itself)
  gcloud compute reservations create "$RES" --zone "$1" --machine-type g2-standard-8 \
    --vm-count 1 --require-specific-reservation --quiet 2>"/tmp/grab_$1.log" && return 0
  grep -qi "accelerator" "/tmp/grab_$1.log" || return 1
  gcloud compute reservations create "$RES" --zone "$1" --machine-type g2-standard-8 \
    --accelerator count=1,type=nvidia-l4 --vm-count 1 --require-specific-reservation --quiet \
    2>"/tmp/grab_$1.log"
}

case "$1" in
  image)
    gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com
    gcloud artifacts repositories describe docsync --location "$REGION" >/dev/null 2>&1 || \
      gcloud artifacts repositories create docsync --repository-format=docker --location "$REGION"
    # Cloud Build builds for linux/amd64, so a Mac's ARM chip doesn't matter
    gcloud builds submit --tag "$REPO/app:v3" --timeout=40m .
    ;;
  grab)
    round=1
    while true; do
      for z in $ZONES; do
        echo ">> round $round, trying $z"
        if reserve "$z"; then
          echo "$z" > /tmp/l4_zone
          echo; echo "L4 RESERVED in $z (billing has started). Next: caffeinate -i bash scripts/gpu/cluster.sh up"
          exit 0
        fi
        tail -1 "/tmp/grab_$z.log"
      done
      echo "all zones full, retrying in 5 min"; round=$((round + 1)); sleep 300
    done
    ;;
  up)
    echo "Zone: $ZONE"
    gcloud services enable container.googleapis.com compute.googleapis.com
    # small CPU node for the app pod
    gcloud container clusters create "$CLUSTER" --zone "$ZONE" \
      --num-nodes 1 --machine-type e2-standard-4 --disk-size 50
    # on-demand L4 node taken from our reservation; GKE installs the NVIDIA driver
    gcloud container node-pools create l4 --cluster "$CLUSTER" --zone "$ZONE" \
      --machine-type g2-standard-8 --num-nodes 1 --disk-size 100 \
      --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
      --reservation-affinity=specific --reservation="$RES"
    gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE"
    kubectl apply -f k8s/week2/hf-cache-pvc.yaml
    kubectl apply -f k8s/week2/docsync-v3.yaml
    kubectl rollout status deploy/docsync --timeout=10m
    kubectl get nodes
    ;;
  down)
    # pods first, then the disk claim (so its disk is deleted), then the cluster, then the reservation
    kubectl delete deploy vllm docsync --ignore-not-found || true
    kubectl delete pvc hf-cache --ignore-not-found --wait=true || true
    # the claim goes first, the disk behind it (its PV) a moment later; deleting the cluster
    # before that leaves the 60 GB disk orphaned and billing, so wait for it
    kubectl wait --for=delete pv --all --timeout=180s || true
    kubectl wait --for=delete pv --all --timeout=180s || true
    gcloud container clusters delete "$CLUSTER" --zone "$ZONE" --quiet || true
    gcloud compute reservations delete "$RES" --zone "$ZONE" --quiet || true
    rm -f /tmp/l4_zone
    bash "$0" check
    ;;
  check)
    for what in "container clusters" "compute instances" "compute reservations" "compute forwarding-rules" "compute disks" "compute addresses"; do
      n=$(gcloud $what list --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
      echo "$what: $n"
      [ "$n" = "0" ] || gcloud $what list
    done
    ;;
  *)
    echo "usage: $0 image|grab|up|down|check"; exit 1 ;;
esac