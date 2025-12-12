from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset import *
from geoseg.models.UNetFormerPP import UNetFormerPP
from tools.utils import Lookahead
from tools.utils import process_model_params

max_epoch = 60
ignore_index = 255
train_batch_size = 1
val_batch_size = 1
lr = 1e-3
weight_decay = 0.05
backbone_lr = 1e-4
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformerpp-r18-1024-768crop-e60-optimized"
weights_path = "model_weights/uavid/{}".format(weights_name)
test_weights_name = "last"
log_name = 'uavid/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = [0]
resume_ckpt_path = None

net = UNetFormerPP(
    num_classes=num_classes,
    use_ecm=True,
    ecm_stages=(2, 3, 4),
    ecm_compression_ratio=2,
    ecm_num_heads=8,
    ecm_window_size=16
)

loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

train_dataset = UAVIDDataset(
    data_root='data/uavid/train_val',
    img_dir='images',
    mask_dir='masks',
    mode='train',
    mosaic_ratio=0.25,
    transform=train_aug,
    img_size=(1024, 1024)
)

val_dataset = UAVIDDataset(
    data_root='data/uavid/val',
    img_dir='images',
    mask_dir='masks',
    mode='val',
    mosaic_ratio=0.0,
    transform=val_aug,
    img_size=(1024, 1024)
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
