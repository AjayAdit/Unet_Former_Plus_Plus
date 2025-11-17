#!/usr/bin/env bash
set -e

# Adjust DATA_ROOT if your teammates store UAVid elsewhere
DATA_ROOT="/extra_space/Downloads/Datasets for DL/D1/dataverse_files/UAVid-v1"
PATCH_ROOT="/extra_space/UAVid_patches"

conda activate airs

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
