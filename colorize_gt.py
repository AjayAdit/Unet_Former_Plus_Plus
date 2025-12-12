import os
import numpy as np
from PIL import Image
from tqdm import tqdm

PALETTE = [
    [128, 0, 0],
    [128, 64, 128],
    [0, 128, 0],
    [128, 128, 0],
    [64, 0, 128],
    [64, 64, 0],
    [0, 64, 64],
    [0, 0, 0],
]

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(PALETTE):
        color_mask[mask == class_id] = color

    color_mask[mask == 255] = [255, 255, 255]

    return color_mask


def colorize_single(input_path, output_path):
    mask = np.array(Image.open(input_path))
    color_mask = colorize_mask(mask)
    Image.fromarray(color_mask).save(output_path)
    print(f"Saved: {output_path}")


def colorize_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    masks = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    print(f"Found {len(masks)} masks")

    for mask_name in tqdm(masks):
        input_path = os.path.join(input_folder, mask_name)
        output_path = os.path.join(output_folder, mask_name)

        mask = np.array(Image.open(input_path))
        color_mask = colorize_mask(mask)
        Image.fromarray(color_mask).save(output_path)

    print(f"\nDone! Colorized masks saved to: {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='data/uavid/val/masks',
                        help='Input mask folder or single file')
    parser.add_argument('--output', '-o', type=str, default='fig_results/uavid/ground_truth',
                        help='Output folder')
    parser.add_argument('--single', '-s', action='store_true',
                        help='Process single file instead of folder')
    args = parser.parse_args()

    if args.single:
        output_path = args.output if args.output.endswith('.png') else args.output + '.png'
        colorize_single(args.input, output_path)
    else:
        colorize_folder(args.input, args.output)
