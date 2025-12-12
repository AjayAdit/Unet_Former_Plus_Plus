"""
Inference script for UNetFormerPP on UAVid patches
Run: python inference_unetformerpp.py

This generates colored prediction masks for visual comparison.
"""

import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add GeoSeg to path
sys.path.insert(0, 'GeoSeg')

from geoseg.models.UNetFormerPP import UNetFormerPP

# ============================================
# CONFIGURATION - UPDATE THESE
# ============================================

# Model checkpoint path (update this!)
CHECKPOINT_PATH = "model_weights/uavid/unetformerpp-r18-1024-768crop-e60-optimized/unetformerpp-r18-1024-768crop-e60-optimized.ckpt"
# Or use: "lightning_logs/uavid/unetformerpp-r18-1024-768crop-e60-optimized/version_2/checkpoints/xxx.ckpt"

# Input/Output paths
INPUT_DIR = "data/uavid/val/images"
OUTPUT_DIR = "fig_results/uavid/unetformerpp/masks"

# Model config (must match training config)
NUM_CLASSES = 8
USE_ECM = True
ECM_STAGES = (2, 3, 4)
ECM_COMPRESSION_RATIO = 2
ECM_NUM_HEADS = 8
ECM_WINDOW_SIZE = 16

# UAVid color palette
PALETTE = np.array([
    [128, 0, 0],      # 0: Building - dark red
    [128, 64, 128],   # 1: Road - purple
    [0, 128, 0],      # 2: Static Car - green
    [128, 128, 0],    # 3: Tree - olive
    [64, 0, 128],     # 4: Low Vegetation - dark purple
    [64, 64, 0],      # 5: Human - dark yellow
    [0, 64, 64],      # 6: Moving Car - teal
    [0, 0, 0],        # 7: Background - black
], dtype=np.uint8)

# ============================================
# FUNCTIONS
# ============================================

def get_transform():
    """Get inference transform"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def colorize_mask(mask):
    """Convert class indices to RGB colors"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(len(PALETTE)):
        color_mask[mask == class_id] = PALETTE[class_id]
    
    return color_mask


def load_model(checkpoint_path):
    """Load UNetFormerPP model from checkpoint"""
    
    # Create model with optimized config
    model = UNetFormerPP(
        num_classes=NUM_CLASSES,
        decode_channels=64,
        pretrained=False,
        use_ecm=USE_ECM,
        ecm_stages=ECM_STAGES,
        ecm_compression_ratio=ECM_COMPRESSION_RATIO,
        ecm_num_heads=ECM_NUM_HEADS,
        ecm_window_size=ECM_WINDOW_SIZE
    )
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'net.' prefix if present (from training wrapper)
            state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please update CHECKPOINT_PATH in the script.")
        sys.exit(1)
    
    return model


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()
    
    # Get transform
    transform = get_transform()
    
    # Get image list
    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    print(f"Found {len(image_paths)} images")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # Load image
            image = np.array(Image.open(img_path).convert('RGB'))
            h, w = image.shape[:2]
            
            # Transform
            transformed = transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            output = model(input_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            # Colorize
            color_mask = colorize_mask(pred)
            
            # Save
            filename = os.path.basename(img_path)
            output_path = os.path.join(OUTPUT_DIR, filename)
            Image.fromarray(color_mask).save(output_path)
    
    print(f"\n✅ Done! Masks saved to: {OUTPUT_DIR}")
    print(f"Total images processed: {len(image_paths)}")


if __name__ == "__main__":
    main()
