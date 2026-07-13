#!/usr/bin/env bash
set -euo pipefail

REPO=/home/chen/Desktop/ActiveSGM
PART_DIR=/tmp/replica_parts
DEST_DIR=$REPO/data/replica_v1_local
LOG=$REPO/logs/replica_download.log

mkdir -p "$PART_DIR" "$DEST_DIR" "$REPO/assets" "$REPO/logs"

for p in {a..q}; do
  f="replica_v1_0.tar.gz.parta${p}"
  url="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/${f}"
  until wget --continue -q "$url" -O "$PART_DIR/$f"; do
    echo "$(date '+%F %T') [RETRY] $f" >> "$LOG"
    sleep 5
  done
  echo "$(date '+%F %T') [OK] $f" >> "$LOG"
done

cat "$PART_DIR"/replica_v1_0.tar.gz.part?? | tar -xzC "$DEST_DIR"

wget -q -O "$REPO/assets/additional_habitat_configs.zip" \
  http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip
unzip -qn "$REPO/assets/additional_habitat_configs.zip" -d "$DEST_DIR"

echo "$(date '+%F %T') [DONE] replica_v1_local ready" >> "$LOG"
