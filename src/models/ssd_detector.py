"""SSD300-VGG16 object detection model implementation, training loop, and inference."""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models import VGG16_Weights
from tqdm import tqdm

from src.constants import CLASS_NAMES_TORCHVISION, TORCH_ID_TO_LABEL
from src.data.torch_dataset import ChessDetectionDataset, collate_fn, get_ssd_transforms


def build_ssd_model(
    num_classes: int = 13,
    pretrained_backbone: bool = True,
    weights_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Build SSD300-VGG16 with custom number of classes (12 pieces + 1 background)."""
    backbone_weights = VGG16_Weights.DEFAULT if pretrained_backbone else None
    model = ssd300_vgg16(
        weights=None,
        num_classes=num_classes,
        weights_backbone=backbone_weights,
    )

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device)
    return model


def train_ssd(
    train_img_dir: str,
    train_lbl_dir: str,
    output_weights: str = "weights/ssd_chess.pth",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    device: Optional[torch.device] = None,
    cache_path: Optional[str] = "cache_ssd_train.pt",
) -> List[float]:
    """Train SSD300 using AdamW optimizer and StepLR scheduler."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_ssd_model(num_classes=13, pretrained_backbone=True, device=device)
    transform = get_ssd_transforms(img_size=300)

    dataset = ChessDetectionDataset(
        img_dir=train_img_dir,
        lbl_dir=train_lbl_dir,
        transform=transform,
        cache_path=cache_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    epoch_losses = []
    print(f"Starting SSD training on {device} for {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            running_loss += loss_val
            loop.set_postfix(loss=f"{loss_val:.4f}")

        scheduler.step()
        avg_loss = running_loss / max(1, len(dataloader))
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_weights)), exist_ok=True)
    torch.save(model.state_dict(), output_weights)
    print(f"SSD model saved to {output_weights}")
    return epoch_losses


def predict_ssd(
    model: torch.nn.Module,
    image_input: Union[str, np.ndarray, Image.Image],
    device: torch.device,
    conf_threshold: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Perform SSD inference on a single image.

    Returns:
        Dictionary with 'boxes' [N, 4], 'scores' [N], 'labels' [N] (labels mapped back to 0..11).
    """
    w_default = VGG16_Weights.DEFAULT
    mean, std = w_default.transforms().mean, w_default.transforms().std

    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(image_input)
    else:
        img_pil = image_input

    orig_w, orig_h = img_pil.size
    resized = img_pil.resize((300, 300))

    img_np = np.array(resized).astype(np.float32) / 255.0
    img_np = (img_np - np.array(mean)) / np.array(std)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        detections = model(tensor)[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()

    # Scale boxes back from 300x300 to original dimensions
    scale_x = orig_w / 300.0
    scale_y = orig_h / 300.0
    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x
    scaled_boxes[:, [1, 3]] *= scale_y

    # Filter by threshold and map labels (sub 1 to convert from 1..12 to 0..11)
    mask = scores >= conf_threshold
    filtered_boxes = scaled_boxes[mask]
    filtered_scores = scores[mask]
    # Class 0 is background; valid pieces are >= 1
    filtered_labels = np.maximum(0, labels[mask] - 1)

    return {
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "labels": filtered_labels,
    }
