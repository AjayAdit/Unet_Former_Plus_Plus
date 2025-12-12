import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, 'GeoSeg')

from geoseg.models.UNetFormerPP import UNetFormerPP

CHECKPOINT_PATH = "model_weights/uavid/unetformerpp-r18-1024-768crop-e60-optimized/unetformerpp-r18-1024-768crop-e60-optimized.ckpt"

INPUT_DIR = "data/uavid/val/images"
OUTPUT_DIR = "fig_results/uavid/unetformerpp/masks"

NUM_CLASSES = 8
USE_ECM = True
ECM_STAGES = (2, 3, 4)
ECM_COMPRESSION_RATIO = 2
ECM_NUM_HEADS = 8
ECM_WINDOW_SIZE = 16

PALETTE = np.array([
    [128, 0, 0],
    [128, 64, 128],
    [0, 128, 0],
    [128, 128, 0],
    [64, 0, 128],
    [64, 64, 0],
    [0, 64, 64],
    [0, 0, 0],
], dtype=np.uint8)


def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id in range(len(PALETTE)):
        color_mask[mask == class_id] = PALETTE[class_id]

    return color_mask


def load_model(checkpoint_path):
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

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)
    model = model.to(device)
    model.eval()

    transform = get_transform()

    image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    print(f"Found {len(image_paths)} images")

    print("Running inference...")
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            image = np.array(Image.open(img_path).convert('RGB'))
            h, w = image.shape[:2]

            transformed = transform(image=image)
            input_tensor = transformed['image'].unsqueeze(0).to(device)

            output = model(input_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

            color_mask = colorize_mask(pred)

            filename = os.path.basename(img_path)
            output_path = os.path.join(OUTPUT_DIR, filename)
            Image.fromarray(color_mask).save(output_path)

    print(f"\n✅ Done! Masks saved to: {OUTPUT_DIR}")
    print(f"Total images processed: {len(image_paths)}")


if __name__ == "__main__":
    main()
