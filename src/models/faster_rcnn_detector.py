"""Faster R-CNN ResNet50-FPN object detection model, training loop, and inference."""

import os
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from src.data.torch_dataset import ChessDetectionDataset, collate_fn, get_faster_rcnn_transforms


def build_faster_rcnn_model(
    num_classes: int = 13,
    pretrained_backbone: bool = True,
    weights_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """Build Faster R-CNN ResNet50-FPN with custom class head (12 pieces + 1 background)."""
    backbone_weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
    )

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model.to(device)
    return model


def train_faster_rcnn(
    train_img_dir: str,
    train_lbl_dir: str,
    output_weights: str = "weights/frcnn_chess.pth",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    device: Optional[torch.device] = None,
    cache_path: Optional[str] = "cache_frcnn_train.pt",
) -> List[float]:
    """Train Faster R-CNN using AdamW optimizer."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_faster_rcnn_model(num_classes=13, pretrained_backbone=True, device=device)
    transform = get_faster_rcnn_transforms(img_size=500)

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
    print(f"Starting Faster R-CNN training on {device} for {epochs} epochs...")

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
    print(f"Faster R-CNN model saved to {output_weights}")
    return epoch_losses


def predict_faster_rcnn(
    model: torch.nn.Module,
    image_input: Union[str, np.ndarray, Image.Image],
    device: torch.device,
    conf_threshold: float = 0.25,
) -> Dict[str, np.ndarray]:
    """Perform Faster R-CNN inference on a single image.

    Returns:
        Dictionary with 'boxes' [N, 4], 'scores' [N], 'labels' [N] (labels mapped back to 0..11).
    """
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img_pil = Image.fromarray(image_input)
    else:
        img_pil = image_input

    tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        detections = model(tensor)[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()

    mask = scores >= conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = np.maximum(0, labels[mask] - 1)

    return {
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "labels": filtered_labels,
    }
