#!/usr/bin/env bash
#
# Pull the entire TomoAnnotator project (code + data, ~12 GB) FROM S3.
# For hackathon participants who have access to the AWS account.
#
# Usage (with your SSO profile):
#   bash pull_from_s3.sh <your-sso-profile> [dest-dir]
# On EC2 with an instance profile attached, omit the profile:
#   bash pull_from_s3.sh "" [dest-dir]
#
# Default destination is ./TomoAnnotator. Uses `aws s3 sync`, so it's
# resumable and only downloads changed/missing files — safe to re-run.

set -euo pipefail

BUCKET="${TOMOANNOTATOR_S3_BUCKET:-scrippsresearch-tomoannotator}"
REGION="${AWS_REGION:-us-west-2}"
PREFIX="${S3_PREFIX:-TomoAnnotator}"
DEST="${2:-./TomoAnnotator}"

PROFILE_ARG=""
if [[ "${1:-}" != "" ]]; then
  PROFILE_ARG="--profile $1"
fi

echo "Source      : s3://$BUCKET/$PREFIX/"
echo "Destination : $DEST"
echo "Region      : $REGION"
echo

mkdir -p "$DEST"
aws s3 sync "s3://$BUCKET/$PREFIX/" "$DEST/" $PROFILE_ARG --region "$REGION"

echo
echo "Done. Then:"
echo "  cd $DEST"
echo "  conda env create -f environment.yml   # or environment-cpu.yml"
echo "  conda activate tomoannotator"
echo "See README.md / PROJECT_JOURNEY.md for the full pipeline."
