"""
4-Image Comparison: Original | Ground Truth | Baseline | UNetFormerPP
Run: python create_comparison.py

Prerequisites:
1. Baseline inference already done: fig_results/uavid/unetformer/masks/
2. Ground truth colorized: fig_results/uavid/ground_truth/
3. UNetFormerPP checkpoint available
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================

# Input paths
ORIGINAL_IMAGES = "data/uavid/val/images"
GROUND_TRUTH = "fig_results/uavid/ground_truth"
BASELINE_MASKS = "fig_results/uavid/unetformer/masks"
UNETFORMERPP_MASKS = "fig_results/uavid/unetformerpp/masks"

# Output path
OUTPUT_DIR = "fig_results/uavid/comparison"

# UAVid color palette (for reference)
UAVID_PALETTE = {
    0: (128, 0, 0),      # Building - dark red
    1: (128, 64, 128),   # Road - purple  
    2: (0, 128, 0),      # Static Car - green
    3: (128, 128, 0),    # Tree - olive
    4: (64, 0, 128),     # Low Vegetation - dark purple
    5: (64, 64, 0),      # Human - dark yellow
    6: (0, 64, 64),      # Moving Car - teal
    7: (0, 0, 0),        # Background - black
}

# ============================================
# FUNCTIONS
# ============================================

def create_single_comparison(original_path, gt_path, baseline_path, ecm_path, output_path):
    """Create a single 4-image comparison"""
    
    # Load images
    original = Image.open(original_path).convert('RGB')
    gt = Image.open(gt_path).convert('RGB')
    baseline = Image.open(baseline_path).convert('RGB')
    ecm = Image.open(ecm_path).convert('RGB')
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gt)
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(baseline)
    axes[2].set_title('Baseline (UNetFormer)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(ecm)
    axes[3].set_title('UNetFormerPP + ECM (Ours)', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_grid(image_names, output_path, rows=3):
    """Create a grid comparison for multiple images"""
    
    n_images = len(image_names)
    
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    
    for i, name in enumerate(image_names[:rows]):
        original = Image.open(os.path.join(ORIGINAL_IMAGES, name)).convert('RGB')
        gt = Image.open(os.path.join(GROUND_TRUTH, name)).convert('RGB')
        baseline = Image.open(os.path.join(BASELINE_MASKS, name)).convert('RGB')
        ecm = Image.open(os.path.join(UNETFORMERPP_MASKS, name)).convert('RGB')
        
        axes[i, 0].imshow(original)
        axes[i, 1].imshow(gt)
        axes[i, 2].imshow(baseline)
        axes[i, 3].imshow(ecm)
        
        for j in range(4):
            axes[i, j].axis('off')
        
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 2].set_title('Baseline', fontsize=12, fontweight='bold')
            axes[i, 3].set_title('ECM (Ours)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved grid: {output_path}")


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if all directories exist
    for path, name in [
        (ORIGINAL_IMAGES, "Original images"),
        (GROUND_TRUTH, "Ground truth"),
        (BASELINE_MASKS, "Baseline masks"),
        (UNETFORMERPP_MASKS, "UNetFormerPP masks")
    ]:
        if not os.path.exists(path):
            print(f"❌ Missing: {name} at {path}")
            print(f"   Please run inference first or check the path.")
            return
        else:
            count = len(glob.glob(os.path.join(path, "*.png")))
            print(f"✅ Found: {name} ({count} images)")
    
    # Get list of images (use baseline as reference)
    image_names = [os.path.basename(f) for f in glob.glob(os.path.join(BASELINE_MASKS, "*.png"))]
    image_names.sort()
    
    print(f"\nTotal images: {len(image_names)}")
    
    # Create individual comparisons
    print("\nCreating individual comparisons...")
    for name in tqdm(image_names[:10]):  # First 10 for speed
        output_path = os.path.join(OUTPUT_DIR, f"comparison_{name}")
        create_single_comparison(
            os.path.join(ORIGINAL_IMAGES, name),
            os.path.join(GROUND_TRUTH, name),
            os.path.join(BASELINE_MASKS, name),
            os.path.join(UNETFORMERPP_MASKS, name),
            output_path
        )
    
    # Create grid comparison (best for report)
    print("\nCreating grid comparison...")
    
    # Select diverse images (different sequences)
    selected = []
    seen_seqs = set()
    for name in image_names:
        seq = name.split('_')[0]  # e.g., "seq16"
        if seq not in seen_seqs:
            selected.append(name)
            seen_seqs.add(seq)
        if len(selected) >= 4:
            break
    
    # If not enough sequences, just take first 4
    if len(selected) < 4:
        selected = image_names[:4]
    
    create_comparison_grid(selected, os.path.join(OUTPUT_DIR, "comparison_grid.png"), rows=len(selected))
    create_comparison_grid(selected, os.path.join(OUTPUT_DIR, "comparison_grid.pdf"), rows=len(selected))
    
    print(f"\n✅ Done! Comparisons saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
