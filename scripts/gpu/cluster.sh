#!/bin/bash
# Create, check and delete the GKE cluster for the week 2 GPU session.
#
#   bash scripts/gpu/cluster.sh image   # build app:v3 with Cloud Build (no GPU, do it the day before)
#   bash scripts/gpu/cluster.sh up      # cluster + on-demand L4 pool + model disk + app pod (~20 min)
#   bash scripts/gpu/cluster.sh down    # delete everything, then run check
#   bash scripts/gpu/cluster.sh check   # list anything still billable; all counts should be 0
set -euo pipefail
PROJECT=${PROJECT:-docsync-vllm-anagha}
ZONE=${ZONE:-us-central1-a}          # if no L4 capacity: ZONE=us-central1-b bash scripts/gpu/cluster.sh up
REGION=us-central1
CLUSTER=docsync
REPO=$REGION-docker.pkg.dev/$PROJECT/docsync

gcloud config set project "$PROJECT" >/dev/null

case "$1" in
  image)
    gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com
    gcloud artifacts repositories describe docsync --location "$REGION" >/dev/null 2>&1 || \
      gcloud artifacts repositories create docsync --repository-format=docker --location "$REGION"
    # Cloud Build builds for linux/amd64, so a Mac's ARM chip doesn't matter
    gcloud builds submit --tag "$REPO/app:v3" --timeout=40m .
    ;;
  up)
    gcloud services enable container.googleapis.com compute.googleapis.com
    # small CPU node for the app pod
    gcloud container clusters create "$CLUSTER" --zone "$ZONE" \
      --num-nodes 1 --machine-type e2-standard-4 --disk-size 50
    # on-demand (not spot) L4 node; GKE installs the NVIDIA driver
    gcloud container node-pools create l4 --cluster "$CLUSTER" --zone "$ZONE" \
      --machine-type g2-standard-8 --num-nodes 1 --disk-size 100 \
      --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest
    gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE"
    kubectl apply -f k8s/week2/hf-cache-pvc.yaml
    kubectl apply -f k8s/week2/docsync-v3.yaml
    kubectl rollout status deploy/docsync --timeout=10m
    kubectl get nodes
    ;;
  down)
    # pods first, then the disk claim (so its disk is deleted), then the cluster
    kubectl delete deploy vllm docsync --ignore-not-found || true
    kubectl delete pvc hf-cache --ignore-not-found --wait=true || true
    gcloud container clusters delete "$CLUSTER" --zone "$ZONE" --quiet
    bash "$0" check
    ;;
  check)
    for what in "container clusters" "compute instances" "compute forwarding-rules" "compute disks" "compute addresses"; do
      n=$(gcloud $what list --format="value(name)" 2>/dev/null | wc -l | tr -d ' ')
      echo "$what: $n"
      [ "$n" = "0" ] || gcloud $what list
    done
    ;;
  *)
    echo "usage: $0 image|up|down|check"; exit 1 ;;
esac