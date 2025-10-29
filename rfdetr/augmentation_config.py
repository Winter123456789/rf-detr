AUG_CONFIG = {
    "train": {
        "ops": [
            {"op": "HorizontalFlip", "p": 0.5},
            {"op": "HueSaturationValue", "hue_shift_limit": int(0.010 * 360), "sat_shift_limit": 0.3, "val_shift_limit": 0.1, "p": 1.0},
            {"op": "Affine", "scale": (0.9, 1.1), "translate_percent": (0.1, 0.1), "rotate": (0, 0), "shear": (0, 0), "p": 0.75},
            # {"op": "Blur", "blur_limit": 7, "p": 0.3},
            # {"op": "GaussNoise", "var_limit": (10.0, 50.0), "p": 0.3},
        ],
        "bbox_params": {"format": "pascal_voc", "min_visibility": 0.01, "clip": True}
    },
    "val": {"ops": []},
    "test": {"ops": []}
}

MOSAIC_CONFIG = {
    "p": 0.5,          
    "grid_yx": (2, 2),
    "fit_mode": "cover" #or "contain"
}
