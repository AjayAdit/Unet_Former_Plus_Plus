# UAVid UNetFormer Baseline (GeoSeg)

This repo contains our UNetFormer baseline on the UAVid-v1 dataset using the GeoSeg framework. 

Clone this repo and follow the steps
- git clone git@github.com:AjayAdit/Unet_Former-.git 
- cd uavid-unetformer-baseline

Data Set Download link:
- https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-x9f-w9sa

## Setup

1. Create the conda environment:
   - conda env create -f environment.yml
   - conda activate airs

2. Download UAVid-v1 and unpack it to, for example:
   /extra_space/Downloads/Datasets for DL/D1/dataverse_files/UAVid-v1

3. Prepare 1024x1024 patches:
   ./scripts/prepare_uavid_patches.sh

4. Train UNetFormer:
   ./scripts/train_unetformer_uavid.sh

The best validation mIoU we obtained was **66.x% at epoch N** (close to the 67% reported in the original UNetFormer paper on UAVid).
