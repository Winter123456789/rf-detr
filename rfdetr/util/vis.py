------------------------------------------------------------------------
# RF-DETR - debug visualization helpers
# ------------------------------------------------------------------------
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def _to_cpu(x):
    return x.detach().cpu() if torch.is_tensor(x) else x

def _denorm(img_chw: torch.Tensor,
            mean: Sequence[float] = IMAGENET_MEAN,
            std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=img_chw.dtype, device=img_chw.device).view(3,1,1)
    std  = torch.tensor(std , dtype=img_chw.dtype, device=img_chw.device).view(3,1,1)
    return (img_chw * std + mean).clamp(0, 1)

def _cxcywh_norm_to_xyxy_abs(boxes_cxcywh: torch.Tensor, size_hw: torch.Tensor) -> torch.Tensor:

    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh
    H, W = size_hw
    scale = torch.tensor([W, H, W, H], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
    c = boxes_cxcywh * scale
    cx, cy, w, h = c.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)

def _draw_boxes(pil_img: Image.Image, boxes_xyxy: torch.Tensor) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return img
    boxes_xyxy = _to_cpu(boxes_xyxy).float()
   
    for i, b in enumerate(boxes_xyxy):
        x0, y0, x1, y1 = [v.item() for v in b]
        draw.rectangle([x0, y0, x1, y1], outline="green", width=1)
    return img

def _safe_image_id(tgt) -> Optional[str]:
    iid = tgt.get("image_id", None)
    if isinstance(iid, torch.Tensor):
        try:
            return str(int(iid.view(-1)[0].item()))
        except Exception:
            return None
    return str(iid) if iid is not None else None

def dump_debug_batch(samples,
                     targets,
                     out_dir: str,
                     epoch: int,
                     step: int,
                     split: str):
    
    imgs = samples.tensors if hasattr(samples, "tensors") else samples
    imgs = _to_cpu(imgs)
    bs = imgs.shape[0]

    out_dir = Path(out_dir) / "debug" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(bs):
        img = _denorm(imgs[i], mean, std)
        pil = F.to_pil_image(img)
        tgt = targets[i]
        size_hw = _to_cpu(tgt["size"]).to(torch.long)
        boxes_cxcywh = _to_cpu(tgt["boxes"]).float() if "boxes" in tgt else torch.zeros((0,4))
        boxes_xyxy = _cxcywh_norm_to_xyxy_abs(boxes_cxcywh, size_hw)
        vis = _draw_boxes(pil, boxes_xyxy)
        iid = _safe_image_id(tgt)
        stem = f"{split}_{step:05d}_img{i:02d}" if iid is None else f"{split}_{step:05d}_id{iid}"
        vis.save(out_dir / f"{stem}.jpg", quality=95)