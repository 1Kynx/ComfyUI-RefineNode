from __future__ import annotations

import torch

from .mask_utils import (
    binary_mask,
    clamp_unit_float,
    mask_batch_size,
    mask_tensor_spatial_size,
    stack_mask_images,
    tensor_mask_to_pil,
)
from .transform_utils import (
    image_batch_size,
    mask_principal_axis_angle,
    normalize_axis_angle,
    normalize_to_srgb,
    rotate_image_canvas,
    rotated_product_crop,
    stack_image_pils,
    tensor_image_to_pil,
)


class RefineNodeMatchProductAngle:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_mask": ("MASK",),
                "reference_mask": ("MASK",),
                "canvas_expand": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("image", "mask", "angle")
    OUTPUT_IS_LIST = (False, False, True)
    FUNCTION = "match_angle"
    CATEGORY = "RefineNode/Transform"

    def match_angle(
        self,
        image: torch.Tensor,
        source_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        canvas_expand: float | int | list[float] | list[int],
    ):
        canvas_expand = clamp_unit_float(canvas_expand, 0.0)
        batch = max(image_batch_size(image), mask_batch_size(source_mask), mask_batch_size(reference_mask))
        reference_size = mask_tensor_spatial_size(reference_mask)
        output_images: list[Image.Image] = []
        output_masks: list[Image.Image] = []
        angles: list[float] = []

        for index in range(batch):
            source_image = normalize_to_srgb(tensor_image_to_pil(image, index))
            source_mask_l = tensor_mask_to_pil(source_mask, index, source_image.size)
            reference_mask_l = tensor_mask_to_pil(reference_mask, index, reference_size)

            source_angle = mask_principal_axis_angle(source_mask_l)
            reference_angle = mask_principal_axis_angle(reference_mask_l)
            if source_angle is None or reference_angle is None:
                output_images.append(source_image)
                output_masks.append(binary_mask(source_mask_l))
                angles.append(0.0)
                continue

            angle = normalize_axis_angle(reference_angle - source_angle)
            rotated_image, rotated_mask = rotated_product_crop(source_image, source_mask_l, angle, canvas_expand)
            output_images.append(rotated_image)
            output_masks.append(rotated_mask)
            angles.append(float(angle))

        return (stack_image_pils(output_images), stack_mask_images(output_masks), angles)


class RefineNodeRotateImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "rotate"
    CATEGORY = "RefineNode/Transform"

    def rotate(
        self,
        image: torch.Tensor,
        angle: float | int | list[float] | list[int],
    ):
        if isinstance(angle, list):
            angle = angle[0] if angle else 0.0
        try:
            angle = float(angle)
        except (TypeError, ValueError):
            angle = 0.0

        output_images = [
            rotate_image_canvas(normalize_to_srgb(tensor_image_to_pil(image, index)), angle)
            for index in range(image_batch_size(image))
        ]
        return (stack_image_pils(output_images),)
