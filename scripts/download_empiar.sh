#!/usr/bin/env bash
# Download tomograms from EMPIAR via FTP (or Aspera if available).
# Usage: ./download_empiar.sh <EMPIAR_ID> <OUTPUT_DIR>
# Example: ./download_empiar.sh 10988 /gpfs/scratch/$USER/TomoAnnotator/raw/

set -euo pipefail

EMPIAR_ID="${1:?Usage: $0 <EMPIAR_ID> <OUTPUT_DIR>}"
OUTPUT_DIR="${2:?Usage: $0 <EMPIAR_ID> <OUTPUT_DIR>}"

mkdir -p "$OUTPUT_DIR/empiar_${EMPIAR_ID}"

FTP_URL="ftp://ftp.ebi.ac.uk/empiar/world_availability/${EMPIAR_ID}/data/"

echo "Downloading EMPIAR-${EMPIAR_ID} to ${OUTPUT_DIR}/empiar_${EMPIAR_ID}/"
echo "FTP URL: ${FTP_URL}"

# Try Aspera first (faster); fall back to wget
if command -v ascp &>/dev/null || module load aspera 2>/dev/null; then
    echo "Using Aspera..."
    ascp -i ~/.aspera/etc/asperaweb_id_dsa.openssh \
         -QT -l 500m -P 33001 \
         "emp_ext2@hx-fasp-1.ebi.ac.uk:/${EMPIAR_ID}/data/" \
         "${OUTPUT_DIR}/empiar_${EMPIAR_ID}/"
else
    echo "Aspera not found, falling back to wget..."
    wget -r -nH --cut-dirs=4 --no-parent -P "${OUTPUT_DIR}/empiar_${EMPIAR_ID}/" "$FTP_URL"
fi

echo "Done. Files in: ${OUTPUT_DIR}/empiar_${EMPIAR_ID}/"
