# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import pycocotools.mask as coco_mask

import rfdetr.datasets.transforms as T
from rfdetr.datasets.transforms import build_albumentations_from_config, ComposeAugmentations, RectResize
from rfdetr.augmentation_config import AUG_CONFIG, MOSAIC_CONFIG
import albumentations as A
import numpy as np
from PIL import Image

def _as_int_resolution(resolution):
    """
    Normalize resolution to integer
    - int -> unchanged
    - (H, W) / [H, W] -> largest side
    """
    if isinstance(resolution, (list, tuple)):
        if len(resolution) != 2:
            raise ValueError(f"resolution must be int or (H, W); got {resolution}")
        return int(max(int(resolution[0]), int(resolution[1])))
    return int(resolution)

def _as_hw(resolution):
    """
    Accept int or (H,W)/[H,W] and returns (H,W).
    """
    if isinstance(resolution, int):
        return resolution, resolution
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    raise TypeError(f"resolution moet int of (H,W) zijn, kreeg: {resolution}")


def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    resolution = _as_int_resolution(resolution)
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, include_masks=False, mosaic_prob = 0.0, mosaic_output_size=None, is_train=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertCoco(include_masks=include_masks)

        self._mosaic_prob = mosaic_prob
        print(mosaic_prob)
        print("shit")
        def _normalize_mosaic_size(x):
            """
            Accepts: None, int, (H,W) tuple/list/torch.Size
            Returns: None or (H, W) as ints
            """
            if x is None:
                return None
            # tuple/list/torch.Size with 2 entries
            if isinstance(x, (list, tuple)) and len(x) == 2:
                return (int(x[0]), int(x[1]))
            try:
                # try generic iterable (e.g., torch.Size)
                it = list(x)
                if len(it) == 2:
                    return (int(it[0]), int(it[1]))
            except Exception:
                pass
            # scalar -> square
            return (int(x), int(x))
        
        self._mosaic_output_hw = _normalize_mosaic_size(mosaic_output_size)
        
        def _sample():
            j = torch.randint(low=0, high=len(self.ids), size=(1,)).item()
            img_j, ann_j = super(CocoDetection, self).__getitem__(j)
            tgt_j = {'image_id': self.ids[j], 'annotations': ann_j}
            return self.prepare(img_j, tgt_j)

        self._mosaic_sampler = _sample


    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # Mosaic (train only, before other augmentations)
        if self._mosaic_prob > 0.0 and self._mosaic_output_hw is not None:
            S_h, S_w = self._mosaic_output_hw
            
            img_np = np.array(img)
            bboxes = target['boxes'].cpu().numpy() if isinstance(target['boxes'], torch.Tensor) else np.array(target['boxes'])
            labels = target['labels'].cpu().tolist() if isinstance(target['labels'], torch.Tensor) else list(target['labels'])

            meta = []
            for _ in range(3):
                im_j, tgt_j = self._mosaic_sampler()
                meta.append({
                    "image": np.array(im_j),
                    "bboxes": tgt_j['boxes'].cpu().numpy() if isinstance(tgt_j['boxes'], torch.Tensor) else np.array(tgt_j['boxes']),
                    "category_ids": tgt_j['labels'].cpu().tolist() if isinstance(tgt_j['labels'], torch.Tensor) else list(tgt_j['labels']),
                })

            gy, gx = MOSAIC_CONFIG.get("grid_yx", (2, 2))
            cell_h = S_h // gy
            cell_w = S_w // gx

            mosaic_tf = A.Compose(
                [
                    A.Mosaic(
                        grid_yx=(gy, gx),
                        cell_shape=(cell_h, cell_w),
                        target_size=(S_h, S_w),
                        metadata_key="mosaic_metadata",
                        fit_mode=MOSAIC_CONFIG.get("fit_mode", "cover"),
                        p=MOSAIC_CONFIG.get("p")
                    ),
                    # force output size, also when Mosaic is skipped due to p-value
                    A.Resize(height=S_h, width=S_w)
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc", label_fields=["category_ids"], clip=True, min_visibility=0.01
                ),
            )

            out = mosaic_tf(image=img_np, bboxes=bboxes, category_ids=labels, mosaic_metadata=meta)
            img = Image.fromarray(out["image"])
            assert img.size == (S_w, S_h), f"Mosaic returned {img.size}, expected {(S_w, S_h)}"
            target = target.copy()

            bxs = out.get('bboxes', [])
            if len(bxs) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.long)
                target['area'] = torch.zeros((0,), dtype=torch.float32)
                target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['boxes'] = torch.as_tensor(bxs, dtype=torch.float32).view(-1, 4)
                target['labels'] = torch.as_tensor(out.get('category_ids', []), dtype=torch.long).view(-1)
                wh = (target['boxes'][:, 2] - target['boxes'][:, 0]) * (target['boxes'][:, 3] - target['boxes'][:, 1])
                target['area'] = wh.to(torch.float32)
                target['iscrowd'] = torch.zeros((target['boxes'].shape[0],), dtype=torch.int64)

            
            target['size'] = torch.tensor([S_h, S_w])
            #target['orig_size'] = torch.tensor([S_h, S_w])
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCoco(object):

    def __init__(self, include_masks=False):
        self.include_masks = include_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    multi-scale disabled
    """
    H, W = _as_hw(resolution)
    print("H, W = ", H, W)
    
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #scales = [_as_int_resolution(resolution)]
    """
    TODO: add multi_scale
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(_as_int_resolution(resolution), expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)
    """
    if image_set == 'train':
        return T.Compose([
            RectResize((H, W)),
            ComposeAugmentations(build_albumentations_from_config(AUG_CONFIG, split="train")),
            normalize,
        ])

    if image_set in ('val', 'test', 'val_speed'):
        return T.Compose([
            RectResize((H, W)),
            normalize,
        ])


    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    """
    return make_coco_transforms(image_set, resolution, multi_scale, expanded_scales, skip_random_resize, patch_size, num_windows)
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    res_int = _as_int_resolution(resolution)
    scales = [res_int]
    if multi_scale:
        # scales = [448, 512, 576, 640, 704, 768, 832, 896]
        scales = compute_multi_scale_scales(res_int, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        print(scales)

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.SquareResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.SquareResize(scales),
                ]),
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.SquareResize([res_int]),
            normalize,
        ])
    if image_set == 'test':
        return T.Compose([
            T.SquareResize([res_int]),
            normalize,
        ])
    if image_set == 'val_speed':
        return T.Compose([
            T.SquareResize([res_int]),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
    """
    
def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root /  "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
        
    mosaic_prob = MOSAIC_CONFIG.get("p", 0.0) if image_set.startswith("train") else 0.0
    print(f"mosaic_prob = {mosaic_prob}")

    
    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
            ),
            mosaic_prob=mosaic_prob, mosaic_output_size=resolution,
            is_train = image_set.startswith('train')                                                                                               
        )
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ),
            mosaic_prob=mosaic_prob, mosaic_output_size=resolution,
            is_train = image_set.startswith('train')                                                                                               
        )
    return dataset

def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
    
    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False

    
    mosaic_prob = MOSAIC_CONFIG.get("p", 0.0) if image_set.startswith("train") else 0.0
    print(f"mosaic_prob = {mosaic_prob}")
    
    if square_resize_div_64:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
            ),
            mosaic_prob=mosaic_prob, mosaic_output_size=resolution,
            is_train = image_set.startswith('train')                                                                                               
        )
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(
            image_set,
            resolution,
            multi_scale=args.multi_scale,
            expanded_scales=args.expanded_scales,
            skip_random_resize=not args.do_random_resize_via_padding,
            patch_size=args.patch_size,
            num_windows=args.num_windows
        ),
            mosaic_prob=mosaic_prob, mosaic_output_size=resolution,
            is_train = image_set.startswith('train')                                                                                               
        )
    return dataset
