from __future__ import annotations

import torch
from typing import Any
from PIL import Image

from .mask_utils import (
    binary_mask,
    clamp_int_value,
    clamp_unit_float,
    flatten_mask_input,
    normalize_choice,
    refine_error,
    slice_masks_by_product_bbox,
    split_mask_components,
    mask_batch_size,
    mask_tensor_spatial_size,
    stack_mask_images,
    tensor_mask_as_list_item,
    tensor_mask_to_pil,
    union_mask_images,
)
from .transform_utils import restore_mask_to_original_space


class RefineNodeMaskBatchProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "combined_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "RefineNode/Mask"

    def process(self, mask: torch.Tensor | list[torch.Tensor], combined_mask: bool | list[bool]):
        combined_mask = bool(combined_mask[0]) if isinstance(combined_mask, list) else bool(combined_mask)
        mask_images = flatten_mask_input(mask)
        size = mask_images[0].size

        if combined_mask:
            return (stack_mask_images([union_mask_images(mask_images, size)]),)

        output_masks = []
        for mask_l in mask_images:
            components = split_mask_components(mask_l)
            if components:
                output_masks.extend(components)

        if not output_masks:
            output_masks = [Image.new("L", size, 0)]

        return (stack_mask_images(output_masks),)


class RefineNodeSliceAndMatchMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_mode": (["mask", "bbox"], {"default": "bbox"}),
                "min_area_ratio": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rows": ("INT", {"default": 4, "min": 1, "max": 16}),
                "columns": ("INT", {"default": 1, "min": 1, "max": 16}),
                "auto_match_orientation": ("BOOLEAN", {"default": True}),
                "match_mode": (["union", "repeat_mask1", "pair_by_index", "pair_by_position"], {"default": "union"}),
            },
            "optional": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask1", "mask2")
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "RefineNode/Mask"

    def process(
        self,
        min_area_ratio: float | list[float],
        rows: int | list[int],
        columns: int | list[int],
        auto_match_orientation: bool | list[bool],
        match_mode: str | list[str],
        output_mode: str | list[str],
        mask1: torch.Tensor | list[torch.Tensor] | None = None,
        mask2: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        min_area_ratio = clamp_unit_float(min_area_ratio, 0.01)
        rows = clamp_int_value(rows, 4, 1, 16)
        columns = clamp_int_value(columns, 1, 1, 16)
        auto_match_orientation = bool(auto_match_orientation[0]) if isinstance(auto_match_orientation, list) else bool(auto_match_orientation)
        match_mode = normalize_choice(match_mode, "union", {"union", "repeat_mask1", "pair_by_index", "pair_by_position"})
        output_mode = normalize_choice(output_mode, "bbox", {"mask", "bbox"})
        if mask1 is None and mask2 is None:
            refine_error("RefineNode Slice And Match Masks", "Connect at least one mask input.")
        if mask1 is None:
            mask1 = mask2
            mask2 = None
        mask1_images = flatten_mask_input(mask1)
        mask2_images = flatten_mask_input(mask2) if mask2 is not None else None
        sliced_mask1, sliced_mask2 = slice_masks_by_product_bbox(
            mask1_images,
            mask2_images,
            min_area_ratio,
            rows,
            columns,
            auto_match_orientation,
            match_mode,
            output_mode,
        )
        return (stack_mask_images(sliced_mask1), stack_mask_images(sliced_mask2))


class RefineNodeRestoreMaskToOriginal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "info": ("REFINENODE_INFO",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "restore"
    CATEGORY = "RefineNode/Mask"

    def restore(self, mask: torch.Tensor | list[torch.Tensor], info: dict[str, Any]):
        mask_inputs = mask if isinstance(mask, list) else [mask]
        info_inputs = info if isinstance(info, list) else [info]

        items = []
        for info_value in info_inputs:
            if isinstance(info_value, dict):
                value_items = info_value.get("items")
                if value_items:
                    items.extend(value_items)
        if not items:
            refine_error("RefineNode Restore Mask To Original", "Missing RefineNode info for mask restore.")

        masks = []
        for mask_input in mask_inputs:
            mask_size = mask_tensor_spatial_size(mask_input)
            for index in range(mask_batch_size(mask_input)):
                masks.append(tensor_mask_to_pil(mask_input, index, mask_size))
        if not masks:
            refine_error("RefineNode Restore Mask To Original", "Missing mask.")

        outputs = []
        output_count = max(len(items), len(masks))
        for item_index in range(output_count):
            item = items[min(item_index, len(items) - 1)]
            if not isinstance(item, dict):
                continue
            mask_l = masks[min(item_index, len(masks) - 1)]
            restored = restore_mask_to_original_space(mask_l, item)
            outputs.append(tensor_mask_as_list_item(restored))

        if not outputs:
            refine_error("RefineNode Restore Mask To Original", "Missing valid RefineNode info items for mask restore.")
        return (outputs,)
