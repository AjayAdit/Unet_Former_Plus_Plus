#!/usr/bin/env bash
set -e
conda activate airs

python GeoSeg/train_supervision.py \
  -c GeoSeg/config/uavid/unetformer.py
