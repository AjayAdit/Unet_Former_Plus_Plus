"""
Simple inference script for UAVid patches
Place this file in: GeoSeg/inference_uavid_patches.py

Usage:
    python GeoSeg/inference_uavid_patches.py \
        -c GeoSeg/config/uavid/unetformer.py \
        -o fig_results/uavid/unetformer
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# UAVid color palette (8 classes)
PALETTE = [
    [128, 0, 0],    # Building - dark red
    [128, 64, 128], # Road - purple
    [0, 128, 0],    # Static Car - green
    [128, 128, 0],  # Tree - olive
    [64, 0, 128],   # Low Vegetation - dark purple
    [64, 64, 0],    # Human - dark yellow
    [0, 64, 64],    # Moving Car - teal
    [0, 0, 0],      # Background - black
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Config file path')
    parser.add_argument('-o', '--output', type=str, default='fig_results/uavid/output', help='Output folder')
    parser.add_argument('--input', type=str, default='data/uavid/val/images', help='Input images folder')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    return parser.parse_args()


def load_config(config_path):
    """Load config file"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def colorize_mask(mask):
    """Convert class mask to RGB color mask"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        color_mask[mask == class_id] = color
    return color_mask


def main():
    args = get_args()
    
    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    
    # Create output folders
    os.makedirs(os.path.join(args.output, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = config.net
    
    # Find checkpoint
    weights_path = config.weights_path
    ckpt_path = os.path.join(weights_path, f"{config.test_weights_name}.ckpt")
    
    if not os.path.exists(ckpt_path):
        # Try to find any checkpoint
        if os.path.exists(weights_path):
            ckpts = [f for f in os.listdir(weights_path) if f.endswith('.ckpt')]
            if ckpts:
                ckpt_path = os.path.join(weights_path, ckpts[0])
                print(f"Using checkpoint: {ckpt_path}")
            else:
                print(f"ERROR: No checkpoint found in {weights_path}")
                print("Please train the model first or check weights_path in config")
                return
        else:
            print(f"ERROR: Weights path does not exist: {weights_path}")
            return
    
    # Load weights
    print(f"Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'net.' prefix if present
        state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # Setup transform
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Get image list
    image_folder = args.input
    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    print(f"Found {len(images)} images in {image_folder}")
    
    if len(images) == 0:
        print("ERROR: No images found!")
        return
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for img_name in tqdm(images):
            # Load image
            img_path = os.path.join(image_folder, img_name)
            img = np.array(Image.open(img_path).convert('RGB'))
            
            # Transform
            transformed = transform(image=img)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Inference
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            
            # Get prediction
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Colorize and save mask
            color_mask = colorize_mask(pred)
            mask_save_path = os.path.join(args.output, 'masks', img_name)
            Image.fromarray(color_mask).save(mask_save_path)
            
            # Copy original image
            img_save_path = os.path.join(args.output, 'images', img_name)
            Image.fromarray(img).save(img_save_path)
    
    print(f"\nDone! Results saved to: {args.output}")
    print(f"  - Predictions: {args.output}/masks/")
    print(f"  - Originals:   {args.output}/images/")


if __name__ == '__main__':
    main()
