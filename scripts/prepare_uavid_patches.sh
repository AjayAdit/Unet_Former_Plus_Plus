# scripts/prepare_uavid_patches.sh

#!/usr/bin/env bash
set -e

DATA_ROOT="/home/ajagan/Unetformer++/Unet_Former_Plus_Plus/dataset_full/UAVid-v1"
PATCH_ROOT="/home/ajagan/Unetformer++/Unet_Former_Plus_Plus/dataset_full/UAVid_patches"

mkdir -p "${PATCH_ROOT}/train/images" "${PATCH_ROOT}/train/masks" \
         "${PATCH_ROOT}/val/images" "${PATCH_ROOT}/val/masks"

python GeoSeg/tools/uavid_patch_split.py \
  --input-dir "${DATA_ROOT}/train" \
  --output-img-dir "${PATCH_ROOT}/train/images" \
  --output-mask-dir "${PATCH_ROOT}/train/masks" \
  --mode 'train' --split-size-h 1024 --split-size-w 1024 \
  --stride-h 1024 --stride-w 1024

python GeoSeg/tools/uavid_patch_split.py \
  --input-dir "${DATA_ROOT}/val" \
  --output-img-dir "${PATCH_ROOT}/val/images" \
  --output-mask-dir "${PATCH_ROOT}/val/masks" \
  --mode 'val' --split-size-h 1024 --split-size-w 1024 \
  --stride-h 1024 --stride-w 1024

