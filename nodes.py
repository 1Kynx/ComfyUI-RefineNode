from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from .mask_utils import (
    RefineEntry,
    bbox_from_mask,
    bbox_from_mask_or_none,
    bbox_mask,
    binary_mask,
    clamp_int_value,
    combine_masks,
    focus_crop,
    make_spatial_mask,
    mask_batch_size,
    mask_indices_for_image,
    normalize_choice,
    offset_bbox,
    prepare_paste_mask,
    refine_error,
    stack_mask_images,
    tensor_mask_as_list_item,
    tensor_mask_to_pil,
)
from .transform_utils import (
    DEFAULT_RESIZE_METHOD,
    IMAGE_RESIZE_METHODS,
    VAE_IMAGE_SIZE,
    calculate_dimensions_area,
    clamp_box,
    composite_masked,
    composite_masked_same_size,
    flatten_refine_info_items,
    flux_kontext_target_size,
    image_batch_size,
    image_content_signature,
    model_paste_mask_to_generated_space,
    normalize_to_srgb,
    pil_to_tensor_image,
    reference_image_transform_metadata,
    restore_generated_to_model_space,
    scale_box,
    safe_size_tuple,
    tensor_image_as_list_item,
    tensor_image_to_pil,
    transformed_content_box,
    update_info_with_kontext_transforms,
    upscale_to_fit_with_padding,
    upscale_to_kontext_size,
)
from .nodes_mask import (
    RefineNodeMaskBatchProcess,
    RefineNodeRestoreMaskToOriginal,
    RefineNodeSliceAndMatchMasks,
)
from .nodes_transform import RefineNodeMatchProductAngle, RefineNodeRotateImage
from .mask_utils import _BILINEAR, _LANCZOS, _NEAREST


def _first_input_value(value: Any, default: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _bool_input_value(value: Any, default: bool = False) -> bool:
    return bool(_first_input_value(value, default))


def _input_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _image_input_count(images: torch.Tensor | list[torch.Tensor] | None) -> int:
    if images is None:
        return 0
    count = 0
    for value in _input_values(images):
        if value is None:
            continue
        count += image_batch_size(value)
    return count


def _mask_input_count(masks: torch.Tensor | list[torch.Tensor] | None) -> int:
    if masks is None:
        return 0
    count = 0
    for value in _input_values(masks):
        if value is None:
            continue
        count += mask_batch_size(value)
    return count


def _select_image_tensor(
    images: torch.Tensor | list[torch.Tensor] | None,
    index: int,
    node_name: str,
    slot_name: str,
) -> torch.Tensor:
    last_value = None
    last_count = 0
    remaining = int(index)
    for value in _input_values(images):
        if value is None:
            continue
        count = image_batch_size(value)
        if count <= 0:
            continue
        if remaining < count:
            return _single_image_tensor(value, remaining, node_name, slot_name)
        remaining -= count
        last_value = value
        last_count = count
    if last_value is not None and last_count > 0:
        return _single_image_tensor(last_value, last_count - 1, node_name, slot_name)
    refine_error(node_name, f"Missing {slot_name} image input.")


def _single_image_tensor(
    image: torch.Tensor,
    index: int,
    node_name: str,
    slot_name: str,
) -> torch.Tensor:
    if image.ndim == 4:
        sample = image[min(int(index), image.shape[0] - 1)].unsqueeze(0)
    elif image.ndim == 3:
        sample = image.unsqueeze(0)
    else:
        refine_error(node_name, f"Expected {slot_name} IMAGE tensor with 3 or 4 dims, got {tuple(image.shape)}.")
    if sample.shape[-1] == 1:
        sample = sample.repeat(1, 1, 1, 3)
    elif sample.shape[-1] == 4:
        sample = sample[..., :3]
    if sample.shape[-1] != 3:
        refine_error(node_name, f"Expected {slot_name} IMAGE channel count 1, 3, or 4, got {sample.shape[-1]}.")
    return sample


def _select_mask_pil(
    masks: torch.Tensor | list[torch.Tensor] | None,
    index: int,
    size: tuple[int, int],
    node_name: str,
) -> Image.Image:
    remaining = int(index)
    for value in _input_values(masks):
        if value is None:
            continue
        count = mask_batch_size(value)
        if count <= 0:
            continue
        if remaining < count:
            return tensor_mask_to_pil(value, remaining, size)
        remaining -= count
    refine_error(node_name, f"Missing mask index {index}.")


class RefineNodePreprocessMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "focus_crop": ("BOOLEAN", {"default": True}),
                "focus_crop_margin": ("INT", {"default": 64, "min": 0, "max": 2048}),
                "spatial_prompt_source": (["mask", "bbox"], {"default": "mask"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "REFINENODE_INFO")
    RETURN_NAMES = ("image", "spatial_mask_image", "mask", "info")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "preprocess"
    CATEGORY = "RefineNode"

    def preprocess(
        self,
        image: torch.Tensor | list[torch.Tensor],
        focus_crop: bool | list[bool],
        focus_crop_margin: int | list[int],
        spatial_prompt_source: str | list[str],
        mask: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        focus_crop_value = _bool_input_value(focus_crop, True)
        focus_crop_margin_value = clamp_int_value(focus_crop_margin, 64, 0, 2048)
        spatial_prompt_source_value = normalize_choice(spatial_prompt_source, "mask", {"mask", "bbox"})
        image_count = _image_input_count(image)
        mask_count = _mask_input_count(mask)
        if image_count <= 0:
            refine_error("RefineNode Preprocess Mask", "Missing input images.")
        model_images = []
        spatial_images = []
        model_masks = []
        infos: list[RefineEntry] = []

        def append_job(
            original: Image.Image,
            mask_l: Image.Image,
            source_image_index: int,
            mask_index: int | None,
            mask_indices: list[int],
        ) -> None:
            bbox_raw = bbox_from_mask_or_none(mask_l)
            has_region = bbox_raw is not None
            model_image = original
            model_mask = mask_l
            crop_box = None
            bbox_model = bbox_raw

            if focus_crop_value and bbox_raw is not None:
                model_image, model_mask, crop_box = focus_crop_region(
                    original,
                    mask_l,
                    bbox_raw,
                    int(focus_crop_margin_value),
                )
                bbox_model = offset_bbox(bbox_raw, crop_box[0], crop_box[1])

            spatial_mask = make_spatial_mask(model_mask, spatial_prompt_source_value, bbox_model)
            group_id = f"source_image_{source_image_index}"

            model_images.append(tensor_image_as_list_item(model_image))
            spatial_images.append(tensor_image_as_list_item(spatial_mask.convert("RGB")))
            model_masks.append(tensor_mask_as_list_item(model_mask))
            infos.append(
                cast(
                    RefineEntry,
                    {
                    "origin_image": original,
                    "origin_size": original.size,
                    "model_image": model_image,
                    "model_mask": model_mask,
                    "spatial_mask": spatial_mask,
                    "bbox_raw": bbox_raw,
                    "bbox_model": bbox_model,
                    "crop_box": crop_box,
                    "has_region": has_region,
                    "spatial_prompt_source": spatial_prompt_source_value,
                    "source_image_index": int(source_image_index),
                    "mask_index": None if mask_index is None else int(mask_index),
                    "mask_indices": [int(value) for value in mask_indices],
                    "component_index": None,
                    "component_count": 0,
                    "group_id": group_id,
                    "combined_mask": False,
                    },
                )
            )

        for image_index in range(image_count):
            original = normalize_to_srgb(
                tensor_image_to_pil(
                    _select_image_tensor(image, image_index, "RefineNode Preprocess Mask", "image"),
                    0,
                )
            )
            if mask is None:
                append_job(
                    original,
                    Image.new("L", original.size, 0),
                    image_index,
                    None,
                    [],
                )
                continue

            indices = mask_indices_for_image(image_index, image_count, mask_count)
            if not indices:
                append_job(
                    original,
                    Image.new("L", original.size, 0),
                    image_index,
                    None,
                    [],
                )
                continue

            for mask_index in indices:
                mask_l = _select_mask_pil(mask, mask_index, original.size, "RefineNode Preprocess Mask")
                append_job(
                    original,
                    mask_l,
                    image_index,
                    mask_index,
                    [mask_index],
                )

        return (
            model_images,
            spatial_images,
            model_masks,
            [{"items": [item]} for item in infos],
        )


class RefineNodeReferenceImageProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "fit_kontext_size": ("BOOLEAN", {"default": True}),
                "resize_method": (IMAGE_RESIZE_METHODS, {"default": "lanczos"}),
                "crop_mode": (["crop", "disable", "fill"], {"default": "crop"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "info": ("REFINENODE_INFO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "REFINENODE_INFO")
    RETURN_NAMES = (
        "image1",
        "image2",
        "image3",
        "info",
    )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "process"
    CATEGORY = "RefineNode"

    def process(
        self,
        image1: torch.Tensor | list[torch.Tensor],
        fit_kontext_size: bool | list[bool],
        resize_method: str | list[str],
        crop_mode: str | list[str],
        image2: torch.Tensor | list[torch.Tensor] | None = None,
        image3: torch.Tensor | list[torch.Tensor] | None = None,
        info: dict[str, Any] | list[Any] | None = None,
    ):
        fit_kontext_size_value = _bool_input_value(fit_kontext_size, True)
        resize_method_value = normalize_choice(resize_method, DEFAULT_RESIZE_METHOD, set(IMAGE_RESIZE_METHODS))
        crop_mode_value = normalize_choice(crop_mode, "crop", {"crop", "disable", "fill"})
        batch = _image_input_count(image1)
        if batch <= 0:
            refine_error("RefineNode Reference Image Process", "Missing image1 input.")

        info_items = flatten_refine_info_items(info)
        if info_items and len(info_items) != batch:
            refine_error(
                "RefineNode Reference Image Process",
                "REFINENODE_INFO item count must match image1 list count; "
                f"got {len(info_items)} info items for {batch} image1 items.",
            )

        image1_outputs = []
        image2_outputs = []
        image3_outputs = []
        info_outputs = []

        for index in range(batch):
            sample1 = _select_image_tensor(image1, index, "RefineNode Reference Image Process", "image1")
            samples1 = tensor_image_to_pil(sample1, 0)
            first = normalize_to_srgb(samples1)
            if fit_kontext_size_value:
                width, height = flux_kontext_target_size(*first.size)
                sizing_mode = "flux_kontext"
            else:
                width, height = calculate_dimensions_area(VAE_IMAGE_SIZE, first.size[0], first.size[1], 8)
                sizing_mode = "area_1024"

            processed = []
            current_transforms: dict[str, list[dict[str, Any] | None]] = {}
            for slot, image in (("image1", image1), ("image2", image2), ("image3", image3)):
                if image is None:
                    current_transforms[slot] = [None]
                    processed.append(sample1.detach().new_zeros((1, height, width, 3)))
                    continue
                sample = _select_image_tensor(image, index, "RefineNode Reference Image Process", slot)
                source_pil = normalize_to_srgb(tensor_image_to_pil(sample, 0))
                current_transforms[slot] = [
                    reference_image_transform_metadata(
                        source_pil.size,
                        (width, height),
                        resize_method_value,
                        crop_mode_value,
                        sizing_mode,
                        image_content_signature(source_pil),
                    )
                ]
                sample = sample.movedim(-1, 1)
                out = upscale_to_kontext_size(
                    sample,
                    width,
                    height,
                    resize_method_value,
                    crop_mode_value,
                ).movedim(1, -1)
                processed.append(out)

            image1_outputs.append(processed[0])
            image2_outputs.append(processed[1])
            image3_outputs.append(processed[2])
            if info_items:
                info_outputs.append(
                    update_info_with_kontext_transforms(
                        {"items": [info_items[index]]},
                        current_transforms,
                    )
                )
            else:
                info_outputs.append({"items": []})

        return (
            image1_outputs,
            image2_outputs,
            image3_outputs,
            info_outputs,
        )


class RefineNodePasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "info": ("REFINENODE_INFO",),
                "paste_back_mode": (["mask", "bbox"], {"default": "mask"}),
                "mask_grow": ("INT", {"default": 3, "min": 0, "max": 256}),
                "blend_blur": ("INT", {"default": 5, "min": 0, "max": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "paste_mask")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "paste_back"
    CATEGORY = "RefineNode"

    def paste_back(
        self,
        generated_image: torch.Tensor,
        info: dict[str, Any],
        paste_back_mode: str,
        mask_grow: int,
        blend_blur: int,
    ):
        generated_inputs = generated_image if isinstance(generated_image, list) else [generated_image]
        info_inputs = info if isinstance(info, list) else [info]
        mode_value = paste_back_mode[0] if isinstance(paste_back_mode, list) else paste_back_mode
        mask_grow_value = mask_grow[0] if isinstance(mask_grow, list) else mask_grow
        blend_blur_value = blend_blur[0] if isinstance(blend_blur, list) else blend_blur

        items = []
        for info_value in info_inputs:
            if isinstance(info_value, dict):
                value_items = info_value.get("items")
                if value_items:
                    items.extend(value_items)
        if not items:
            refine_error("RefineNode Paste Back", "Missing RefineNode Preprocess Mask info.")

        generated_images = []
        for generated_input in generated_inputs:
            for index in range(image_batch_size(generated_input)):
                generated_images.append(normalize_to_srgb(tensor_image_to_pil(generated_input, index)))
        if not generated_images:
            refine_error("RefineNode Paste Back", "Missing generated images.")
        if len(generated_images) < len(items):
            refine_error(
                "RefineNode Paste Back",
                "requires one generated image per RefineNode info item; "
                f"got {len(generated_images)} generated images for {len(items)} items.",
            )

        mode = (mode_value or "mask").strip().lower()
        groups: dict[str, dict[str, Any]] = {}
        group_order = []
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            group_id = item.get("group_id")
            if group_id is None:
                group_id = f"source_image_{item.get('source_image_index', item_index)}"
            group_id = str(group_id)
            if group_id not in groups:
                groups[group_id] = {
                    "origin_image": item["origin_image"],
                    "entries": [],
                }
                group_order.append(group_id)
            groups[group_id]["entries"].append((item_index, item))
        if not group_order:
            refine_error("RefineNode Paste Back", "Missing valid RefineNode Preprocess Mask info items.")

        outputs = []
        masks = []
        for group_id in group_order:
            group = groups[group_id]
            result = group["origin_image"].copy()
            combined_mask = Image.new("L", result.size, 0)

            for item_index, item in group["entries"]:
                generated = generated_images[item_index]
                result, full_mask = self.apply_item_to_result(
                    result,
                    item,
                    generated,
                    mode,
                    int(mask_grow_value),
                    int(blend_blur_value),
                )
                combined_mask = Image.fromarray(
                    np.maximum(
                        np.asarray(combined_mask.convert("L"), dtype=np.uint8),
                        np.asarray(self.mask_for_size(full_mask, combined_mask.size), dtype=np.uint8),
                    ),
                    mode="L",
                )

            outputs.append(tensor_image_as_list_item(result))
            masks.append(tensor_mask_as_list_item(combined_mask))

        return (outputs, masks)

    def apply_item_to_result(
        self,
        current_result: Image.Image,
        item: dict[str, Any],
        generated: Image.Image,
        mode: str,
        mask_grow: int,
        blend_blur: int,
    ) -> tuple[Image.Image, Image.Image]:
        original = item.get("origin_image")
        model_image = item.get("model_image")
        model_mask = item.get("model_mask")
        group_id = str(item.get("group_id") or "")
        if not isinstance(original, Image.Image):
            refine_error("RefineNode Paste Back", "Missing original image.", group_id=group_id)
        if not isinstance(model_image, Image.Image):
            refine_error("RefineNode Paste Back", "Missing model image.", group_id=group_id)
        if not isinstance(model_mask, Image.Image):
            refine_error("RefineNode Paste Back", "Missing model mask.", group_id=group_id)
        crop_box = item.get("crop_box")
        bbox_model = item.get("bbox_model")
        has_region = bool(item.get("has_region", bbox_model is not None))
        generated = restore_generated_to_model_space(
            generated,
            model_image,
            item.get("reference_image_transform"),
        )

        if not has_region or bbox_model is None:
            paste_mask = Image.new("L", model_image.size, 255)
        elif mode == "bbox":
            paste_mask = bbox_mask(model_image.size, bbox_model)
        else:
            paste_mask = binary_mask(model_mask)
        paste_mask = prepare_paste_mask(paste_mask, int(mask_grow), int(blend_blur))

        if crop_box:
            result = current_result.copy()
            current_crop = result.crop(crop_box)
            result_crop = composite_masked(current_crop, generated, paste_mask)
            result.paste(result_crop, (crop_box[0], crop_box[1]))
            full_mask = Image.new("L", original.size, 0)
            full_mask.paste(self.mask_for_size(paste_mask, result_crop.size), (crop_box[0], crop_box[1]))
            return result, full_mask

        result = composite_masked(current_result, generated, paste_mask)
        full_mask = self.mask_for_size(paste_mask, current_result.size)
        return result, full_mask

    def mask_for_size(self, mask_l: Image.Image, size: tuple[int, int]) -> Image.Image:
        mask_l = mask_l.convert("L")
        if mask_l.size == size:
            return mask_l
        return mask_l.resize(size, _BILINEAR)


class RefineNodeMergeGeneratedImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "info": ("REFINENODE_INFO",),
                "mask_source": (["mask", "bbox"], {"default": "mask"}),
                "mask_grow": ("INT", {"default": 3, "min": 0, "max": 256}),
                "blend_blur": ("INT", {"default": 5, "min": 0, "max": 256}),
                "show_full_image": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "paste_mask")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "merge"
    CATEGORY = "RefineNode"

    def merge(
        self,
        generated_image: torch.Tensor,
        info: dict[str, Any],
        mask_source: str = "mask",
        mask_grow: int = 3,
        blend_blur: int = 5,
        show_full_image: bool = False,
        paste_back_mode: str | None = None,
    ):
        generated_inputs = generated_image if isinstance(generated_image, list) else [generated_image]
        info_inputs = info if isinstance(info, list) else [info]
        legacy_mode = paste_back_mode[0] if isinstance(paste_back_mode, list) else paste_back_mode
        new_mode = mask_source[0] if isinstance(mask_source, list) else mask_source
        mode_value = legacy_mode if legacy_mode is not None and (new_mode is None or new_mode == "mask") else new_mode
        mask_grow_value = mask_grow[0] if isinstance(mask_grow, list) else mask_grow
        blend_blur_value = blend_blur[0] if isinstance(blend_blur, list) else blend_blur
        show_full_image_value = bool(show_full_image[0]) if isinstance(show_full_image, list) else bool(show_full_image)

        items = []
        for info_value in info_inputs:
            if isinstance(info_value, dict):
                value_items = info_value.get("items")
                if value_items:
                    items.extend(value_items)
        if not items:
            refine_error("RefineNode Merge Generated Images", "Missing RefineNode Preprocess Mask info.")

        generated_images = []
        for generated_input in generated_inputs:
            for index in range(image_batch_size(generated_input)):
                generated_images.append(normalize_to_srgb(tensor_image_to_pil(generated_input, index)))
        if not generated_images:
            refine_error("RefineNode Merge Generated Images", "Missing generated images.")
        if len(generated_images) < len(items):
            refine_error(
                "RefineNode Merge Generated Images",
                "requires one generated image per RefineNode info item; "
                f"got {len(generated_images)} generated images for {len(items)} items.",
            )

        mode = (mode_value or "mask").strip().lower()
        groups: dict[str, dict[str, Any]] = {}
        group_order = []
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            group_id = item.get("group_id")
            if group_id is None:
                group_id = f"source_image_{item.get('source_image_index', item_index)}"
            group_id = str(group_id)
            generated = generated_images[item_index]
            if group_id not in groups:
                groups[group_id] = {"entries": []}
                group_order.append(group_id)
            groups[group_id]["entries"].append((item_index, item, generated))
        if not group_order:
            refine_error("RefineNode Merge Generated Images", "Missing valid RefineNode Preprocess Mask info items.")

        outputs = []
        masks = []
        for group_id in group_order:
            group = groups[group_id]
            entries = group["entries"]
            if not entries:
                continue
            output_image, output_mask = self.merge_group(
                entries,
                mode,
                int(mask_grow_value),
                int(blend_blur_value),
                show_full_image_value,
            )

            outputs.append(tensor_image_as_list_item(output_image))
            masks.append(tensor_mask_as_list_item(output_mask))

        return (outputs, masks)

    def merge_group(
        self,
        entries: list[tuple[int, dict[str, Any], Image.Image]],
        mode: str,
        mask_grow: int,
        blend_blur: int,
        show_full_image: bool,
    ) -> tuple[Image.Image, Image.Image]:
        origin = self.original_image_for_item(entries[0][1])
        geometries = [
            self.entry_geometry(item, generated.size, origin.size)
            for _, item, generated in entries
        ]
        scales = [geometry[0] for geometry in geometries]
        scale = float(np.median(scales)) if scales else 1.0
        scale = max(scale, 1e-6)
        canvas_size = (
            max(1, int(round(origin.size[0] * scale))),
            max(1, int(round(origin.size[1] * scale))),
        )
        result = origin.resize(canvas_size, _LANCZOS)
        combined_mask = Image.new("L", canvas_size, 0)
        patch_rects: list[tuple[int, int, int, int]] = []

        for (_, item, generated), (_, origin_offset, generated_box, live_origin_size) in zip(entries, geometries):
            paste_mask = self.paste_mask_for_item(
                item,
                generated.size,
                mode,
                int(mask_grow),
                int(blend_blur),
            )
            live_generated = generated.crop(generated_box)
            live_mask = paste_mask.crop(generated_box)
            target_size = (
                max(1, int(round(float(live_origin_size[0]) * scale))),
                max(1, int(round(float(live_origin_size[1]) * scale))),
            )
            if live_generated.size != target_size:
                live_generated = live_generated.resize(target_size, _LANCZOS)
                live_mask = live_mask.resize(target_size, _BILINEAR)

            paste_x = int(round(origin_offset[0] * scale))
            paste_y = int(round(origin_offset[1] * scale))
            patch_rect = self.composite_generated_patch(
                result,
                combined_mask,
                live_generated,
                live_mask,
                paste_x,
                paste_y,
            )
            if patch_rect is not None:
                patch_rects.append(patch_rect)

        if show_full_image:
            return result, combined_mask

        output_rect = bbox_from_mask_or_none(combined_mask)
        if output_rect is None:
            output_rect = self.union_rects(patch_rects)
        if output_rect is None:
            output_rect = (0, 0, result.size[0], result.size[1])
        output_rect = clamp_box(output_rect, result.size)
        return result.crop(output_rect), combined_mask.crop(output_rect)

    def entry_geometry(
        self,
        item: dict[str, Any],
        generated_size: tuple[int, int],
        origin_size: tuple[int, int],
    ) -> tuple[float, tuple[float, float], tuple[int, int, int, int], tuple[float, float]]:
        crop_box = self.item_crop_box(item)
        if crop_box is None:
            crop_box = (0, 0, origin_size[0], origin_size[1])
        cx1, cy1, cx2, cy2 = crop_box
        crop_w = max(1, int(cx2) - int(cx1))
        crop_h = max(1, int(cy2) - int(cy1))
        gen_w, gen_h = generated_size
        transform = item.get("reference_image_transform")

        if not isinstance(transform, dict):
            scale = math.sqrt(float(gen_w * gen_h) / float(crop_w * crop_h))
            return max(scale, 1e-6), (float(cx1), float(cy1)), (0, 0, gen_w, gen_h), (float(crop_w), float(crop_h))

        mode = (transform.get("crop_mode") or "crop").strip().lower()
        model_image = item.get("model_image")
        model_size = model_image.size if isinstance(model_image, Image.Image) else (crop_w, crop_h)
        source_size = safe_size_tuple(transform.get("source_size"), model_size)
        target_size = safe_size_tuple(transform.get("target_size"), generated_size)
        source_w, source_h = source_size
        source_to_origin_x = float(crop_w) / float(source_w)
        source_to_origin_y = float(crop_h) / float(source_h)

        if mode == "crop":
            source_box = transformed_content_box(
                transform,
                "source_content_box",
                source_size,
                source_size,
                (0, 0, source_w, source_h),
            )
            sx1, sy1, sx2, sy2 = source_box
            live_src_w = max(1, sx2 - sx1)
            live_src_h = max(1, sy2 - sy1)
            scale = float(gen_w) / float(live_src_w)
            offset_x = float(cx1) + float(sx1) * source_to_origin_x
            offset_y = float(cy1) + float(sy1) * source_to_origin_y
            live_origin_size = (
                float(live_src_w) * source_to_origin_x,
                float(live_src_h) * source_to_origin_y,
            )
            return max(scale, 1e-6), (offset_x, offset_y), (0, 0, gen_w, gen_h), live_origin_size

        if mode == "fill":
            generated_box = transformed_content_box(
                transform,
                "target_content_box",
                target_size,
                generated_size,
                (0, 0, target_size[0], target_size[1]),
            )
            gx1, gy1, gx2, gy2 = generated_box
            live_w = max(1, gx2 - gx1)
            scale = float(live_w) / float(crop_w)
            return max(scale, 1e-6), (float(cx1), float(cy1)), generated_box, (float(crop_w), float(crop_h))

        sx = float(gen_w) / float(source_w)
        sy = float(gen_h) / float(source_h)
        scale = (max(sx, 1e-6) + max(sy, 1e-6)) / 2.0
        if abs(sx - sy) / max(abs(sx), abs(sy), 1e-6) > 0.02:
            print(
                "RefineNode Merge Generated Images: crop_mode=disable uses anisotropic source scaling "
                f"({sx:.3f}x vs {sy:.3f}y); resampling the patch to fit the shared group canvas."
            )
        return max(scale, 1e-6), (float(cx1), float(cy1)), (0, 0, gen_w, gen_h), (float(crop_w), float(crop_h))

    def original_image_for_item(self, item: dict[str, Any]) -> Image.Image:
        origin_image = item.get("origin_image")
        if isinstance(origin_image, Image.Image):
            return origin_image.convert("RGB")
        model_image = item.get("model_image")
        if isinstance(model_image, Image.Image):
            return model_image.convert("RGB")
        refine_error(
            "RefineNode Merge Generated Images",
            "Missing original/model image for generated image merge.",
            group_id=str(item.get("group_id") or ""),
        )

    def composite_generated_patch(
        self,
        result: Image.Image,
        combined_mask: Image.Image,
        generated: Image.Image,
        paste_mask: Image.Image,
        paste_x: int,
        paste_y: int,
    ) -> tuple[int, int, int, int] | None:
        canvas_w, canvas_h = result.size
        src_w, src_h = generated.size
        dst_x1 = max(0, int(paste_x))
        dst_y1 = max(0, int(paste_y))
        dst_x2 = min(canvas_w, int(paste_x) + src_w)
        dst_y2 = min(canvas_h, int(paste_y) + src_h)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            return None

        src_x1 = dst_x1 - int(paste_x)
        src_y1 = dst_y1 - int(paste_y)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        dst_box = (dst_x1, dst_y1, dst_x2, dst_y2)
        src_box = (src_x1, src_y1, src_x2, src_y2)

        destination = result.crop(dst_box)
        source = generated.crop(src_box)
        mask_crop = paste_mask.convert("L").crop(src_box)
        result_crop = composite_masked_same_size(destination, source, mask_crop)
        result.paste(result_crop, (dst_x1, dst_y1))

        current_mask = np.asarray(combined_mask.crop(dst_box).convert("L"), dtype=np.uint8)
        patch_mask = np.asarray(mask_crop.convert("L"), dtype=np.uint8)
        merged_mask = Image.fromarray(np.maximum(current_mask, patch_mask), mode="L")
        combined_mask.paste(merged_mask, (dst_x1, dst_y1))
        return dst_box

    def union_rects(self, rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
        valid = [rect for rect in rects if rect[2] > rect[0] and rect[3] > rect[1]]
        if not valid:
            return None
        return (
            min(rect[0] for rect in valid),
            min(rect[1] for rect in valid),
            max(rect[2] for rect in valid),
            max(rect[3] for rect in valid),
        )

    def item_crop_box(self, item: dict[str, Any]) -> tuple[int, int, int, int] | None:
        crop_box = item.get("crop_box")
        if isinstance(crop_box, (list, tuple)) and len(crop_box) == 4:
            try:
                x1, y1, x2, y2 = (int(value) for value in crop_box)
            except (TypeError, ValueError):
                return None
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2
        return None

    def model_image_for_item(self, item: dict[str, Any]) -> Image.Image:
        model_image = item.get("model_image")
        if isinstance(model_image, Image.Image):
            return model_image.convert("RGB")
        origin_image = item.get("origin_image")
        if isinstance(origin_image, Image.Image):
            return origin_image.convert("RGB")
        refine_error(
            "RefineNode Merge Generated Images",
            "Missing model/original image for generated-space merge.",
            group_id=str(item.get("group_id") or ""),
        )

    def base_model_paste_mask_for_item(self, item: dict[str, Any], mode: str) -> Image.Image:
        model_image = self.model_image_for_item(item)
        model_size = model_image.size
        bbox_model = item.get("bbox_model")
        has_region = bool(item.get("has_region", bbox_model is not None))
        if not has_region or bbox_model is None:
            return Image.new("L", model_size, 255)

        if (mode or "mask").strip().lower() == "bbox":
            return bbox_mask(model_size, bbox_model)

        model_mask = item.get("model_mask")
        if isinstance(model_mask, Image.Image):
            return binary_mask(model_mask).resize(model_size, _NEAREST)
        return Image.new("L", model_size, 255)

    def background_for_item(self, item: dict[str, Any], generated_size: tuple[int, int]) -> Image.Image:
        model_image = item.get("model_image")
        if not isinstance(model_image, Image.Image):
            model_image = item.get("origin_image")
        if not isinstance(model_image, Image.Image):
            refine_error(
                "RefineNode Merge Generated Images",
                "Missing model/original image for generated-space merge.",
                group_id=str(item.get("group_id") or ""),
            )
        return model_image_to_generated_space(
            model_image,
            generated_size,
            item.get("reference_image_transform"),
        )

    def paste_mask_for_item(
        self,
        item: dict[str, Any],
        generated_size: tuple[int, int],
        mode: str,
        mask_grow: int,
        blend_blur: int,
    ) -> Image.Image:
        model_paste_mask = self.base_model_paste_mask_for_item(item, mode)
        prepared_model_mask = prepare_paste_mask(model_paste_mask, int(mask_grow), int(blend_blur))
        return model_paste_mask_to_generated_space(
            prepared_model_mask,
            generated_size,
            item.get("reference_image_transform"),
        )


def focus_crop_region(
    image: Image.Image,
    mask_l: Image.Image,
    bbox: tuple[int, int, int, int],
    margin: int,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    return focus_crop(image, mask_l, bbox, margin)

NODE_CLASS_MAPPINGS = {
    "RefineNodeMaskBatchProcess": RefineNodeMaskBatchProcess,
    "RefineNodeSliceAndMatchMasks": RefineNodeSliceAndMatchMasks,
    "RefineNodeMatchProductAngle": RefineNodeMatchProductAngle,
    "RefineNodeRotateImage": RefineNodeRotateImage,
    "RefineNodePreprocessMask": RefineNodePreprocessMask,
    "RefineNodeReferenceImageProcess": RefineNodeReferenceImageProcess,
    "RefineNodeRestoreMaskToOriginal": RefineNodeRestoreMaskToOriginal,
    "RefineNodePasteBack": RefineNodePasteBack,
    "RefineNodeMergeGeneratedImages": RefineNodeMergeGeneratedImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RefineNodeMaskBatchProcess": "RefineNode Mask Batch Process",
    "RefineNodeSliceAndMatchMasks": "RefineNode Slice And Match Masks",
    "RefineNodeMatchProductAngle": "RefineNode Match Product Angle",
    "RefineNodeRotateImage": "RefineNode Rotate Image",
    "RefineNodePreprocessMask": "RefineNode Preprocess Mask",
    "RefineNodeReferenceImageProcess": "RefineNode Reference Image Process",
    "RefineNodeRestoreMaskToOriginal": "RefineNode Restore Mask To Original",
    "RefineNodePasteBack": "RefineNode Paste Back",
    "RefineNodeMergeGeneratedImages": "RefineNode Merge Generated Images",
}
