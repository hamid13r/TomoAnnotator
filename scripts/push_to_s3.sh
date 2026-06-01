#!/usr/bin/env bash
#
# Push the entire TomoAnnotator project (code + data, ~12 GB) to S3 so other
# hackathon participants (who have access to the AWS account) can pull it.
#
# Run this from the project root, on a machine where your AWS creds work:
#   bash scripts/push_to_s3.sh <your-sso-profile>
# On EC2 with the instance profile attached, omit the profile:
#   bash scripts/push_to_s3.sh
#
# Uses `aws s3 sync`, which is resumable and only re-uploads changed files,
# so it's safe to re-run.

set -euo pipefail

BUCKET="${TOMOANNOTATOR_S3_BUCKET:-scrippsresearch-tomoannotator}"
REGION="${AWS_REGION:-us-west-2}"
PREFIX="${S3_PREFIX:-TomoAnnotator}"          # s3://<bucket>/TomoAnnotator/...
PROFILE_ARG=""
if [[ "${1:-}" != "" ]]; then
  PROFILE_ARG="--profile $1"
fi

# Resolve project root = parent of this script's dir, so it works from anywhere.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Project root : $ROOT"
echo "Destination  : s3://$BUCKET/$PREFIX/"
echo "Region       : $REGION"
echo

# Create the bucket if it doesn't exist (no-op if it already does).
if ! aws s3api head-bucket --bucket "$BUCKET" $PROFILE_ARG --region "$REGION" 2>/dev/null; then
  echo "Bucket not found — creating s3://$BUCKET/ ..."
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION" $PROFILE_ARG
fi

# Sync everything. Exclude things nobody needs:
#   .git history, Python caches, OS cruft.
aws s3 sync . "s3://$BUCKET/$PREFIX/" \
  $PROFILE_ARG --region "$REGION" \
  --exclude ".git/*" \
  --exclude "*/__pycache__/*" \
  --exclude "*.pyc" \
  --exclude ".DS_Store"

echo
echo "Done. Others with account access can pull the whole project with:"
echo "  aws s3 sync s3://$BUCKET/$PREFIX/ ./TomoAnnotator/ --region $REGION"
