#!/usr/bin/env bash
set -e

python GeoSeg/train_supervision.py \
  -c GeoSeg/config/uavid/unetformer.py
