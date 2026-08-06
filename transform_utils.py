from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageCms

from .mask_utils import (
    _BICUBIC,
    _BILINEAR,
    _NEAREST,
    _LANCZOS,
    _BOX,
    bbox_from_mask,
    bbox_from_mask_or_none,
    binary_mask,
    bbox_mask,
    prepare_paste_mask,
    refine_error,
)


VAE_IMAGE_SIZE = 1024 * 1024
DEFAULT_RESIZE_METHOD = "lanczos"
IMAGE_RESIZE_METHODS = ["lanczos", "area", "bicubic", "bilinear", "nearest-exact"]
SUPPORTED_UPSCALE_METHODS = {"lanczos", "area", "bicubic", "bilinear", "nearest-exact"}
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def calculate_dimensions_rounded(target_area: int, ratio: float, multiple: int) -> tuple[int, int]:
    width = math.sqrt(float(target_area) * float(ratio))
    height = width / float(ratio)
    width = max(multiple, round(width / multiple) * multiple)
    height = max(multiple, round(height / multiple) * multiple)
    return int(width), int(height)


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    return calculate_dimensions_rounded(target_area, ratio, 32)


def calculate_dimensions_area(target_area: int, width: int, height: int, multiple: int | None = None) -> tuple[int, int]:
    scale_by = math.sqrt(float(target_area) / float(width * height))
    target_width = width * scale_by
    target_height = height * scale_by
    if multiple is not None:
        target_width = round(target_width / multiple) * multiple
        target_height = round(target_height / multiple) * multiple
        return max(multiple, int(target_width)), max(multiple, int(target_height))
    return max(1, round(target_width)), max(1, round(target_height))


def flux_kontext_target_size(width: int, height: int) -> tuple[int, int]:
    aspect_ratio = width / height
    _, target_width, target_height = min(
        (abs(aspect_ratio - w / h), w, h)
        for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return target_width, target_height


def normalize_to_srgb(img: Image.Image | None) -> Image.Image | None:
    if img is None:
        return None
    icc = img.info.get("icc_profile") if hasattr(img, "info") else None
    if icc:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst_profile = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(
                img,
                src_profile,
                dst_profile,
                outputMode="RGB",
            )
        except Exception:
            return img.convert("RGB")
    return img.convert("RGB")


def composite_masked(
    destination: Image.Image,
    source: Image.Image,
    mask_l: Image.Image,
    *,
    allow_resize: bool = False,
) -> Image.Image:
    if source.size != destination.size:
        if not allow_resize:
            raise ValueError(
                f"RefineNode composite size mismatch: source size {source.size} "
                f"does not match destination size {destination.size}."
            )
        source = source.resize(destination.size, _BICUBIC)
    if mask_l.size != destination.size:
        if not allow_resize:
            raise ValueError(
                f"RefineNode composite size mismatch: mask size {mask_l.size} "
                f"does not match destination size {destination.size}."
            )
        mask_l = mask_l.resize(destination.size, _BILINEAR)

    dst = np.asarray(destination.convert("RGB"), dtype=np.float32)
    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask_l.convert("L"), dtype=np.float32) / 255.0
    out = src * alpha[:, :, None] + dst * (1.0 - alpha[:, :, None])
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), mode="RGB")


def composite_masked_same_size(
    destination: Image.Image,
    source: Image.Image,
    mask_l: Image.Image,
) -> Image.Image:
    return composite_masked(destination, source, mask_l, allow_resize=False)


def tensor_image_to_pil(images: torch.Tensor, index: int) -> Image.Image:
    tensor = images.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[min(index, tensor.shape[0] - 1)]
    if tensor.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got {tuple(images.shape)}")
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
    elif tensor.shape[-1] == 4:
        tensor = tensor[..., :3]
    if tensor.shape[-1] != 3:
        raise ValueError(f"Expected IMAGE tensor channel count 1, 3, or 4, got {tensor.shape[-1]}")
    arr = (tensor.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor_image(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def image_content_signature(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    thumb = image.convert("RGB").resize((32, 32), _BILINEAR)
    return hashlib.sha1(thumb.tobytes()).hexdigest()


def image_batch_size(images: torch.Tensor | None) -> int:
    if images is None:
        return 0
    if images.ndim == 4:
        return int(images.shape[0])
    return 1


def tensor_image_as_list_item(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor_image(image).unsqueeze(0)


def normalize_axis_angle(angle: float) -> float:
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return angle


def mask_principal_axis_angle(mask_l: Image.Image) -> float | None:
    arr = np.asarray(mask_l.convert("L"), dtype=np.uint8) > 0
    ys, xs = np.nonzero(arr)
    if xs.size < 2:
        return None
    coords = np.stack([xs.astype(np.float64), -ys.astype(np.float64)], axis=1)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords, rowvar=False)
    if not np.isfinite(cov).all():
        return None
    values, vectors = np.linalg.eigh(cov)
    vector = vectors[:, int(np.argmax(values))]
    if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-8:
        return None
    return normalize_axis_angle(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def rotated_product_crop(
    image: Image.Image,
    mask_l: Image.Image,
    angle: float,
    canvas_expand: float = 0.0,
) -> tuple[Image.Image, Image.Image]:
    rotated_image = image.convert("RGB").rotate(
        angle,
        resample=_BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )
    rotated_mask = binary_mask(mask_l).rotate(
        angle,
        resample=_NEAREST,
        expand=True,
        fillcolor=0,
    )
    rotated_mask = binary_mask(rotated_mask)
    bbox = bbox_from_mask_or_none(rotated_mask)
    if bbox is None:
        return rotated_image, rotated_mask
    expand = max(0.0, min(1.0, float(canvas_expand)))
    if expand <= 0.0:
        crop_box = bbox
    else:
        canvas_w, canvas_h = rotated_image.size
        x1, y1, x2, y2 = bbox
        crop_box = (
            max(0, int(math.floor(x1 * (1.0 - expand)))),
            max(0, int(math.floor(y1 * (1.0 - expand)))),
            min(canvas_w, int(math.ceil(x2 * (1.0 - expand) + canvas_w * expand))),
            min(canvas_h, int(math.ceil(y2 * (1.0 - expand) + canvas_h * expand))),
        )
    return rotated_image.crop(crop_box), rotated_mask.crop(crop_box)


def rotate_image_canvas(image: Image.Image, angle: float) -> Image.Image:
    return image.convert("RGB").rotate(
        float(angle),
        resample=_BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0),
    )


def stack_image_pils(images: list[Image.Image]) -> torch.Tensor:
    if not images:
        raise ValueError("Cannot output an empty image batch.")
    size = images[0].size
    if any(image.size != size for image in images):
        raise ValueError(
            "RefineNode Match Product Angle batch outputs have different crop sizes. "
            "Split the batch or use inputs with matching rotated product bbox sizes."
        )
    return torch.stack([pil_to_tensor_image(image) for image in images], dim=0)


def upscale_image(samples: torch.Tensor, width: int, height: int, method: str, crop: str = "disabled") -> torch.Tensor:
    import comfy.utils

    method = (method or DEFAULT_RESIZE_METHOD).strip().lower()
    if method == "nearest":
        method = "nearest-exact"
    if method not in SUPPORTED_UPSCALE_METHODS:
        method = "bicubic"
    return comfy.utils.common_upscale(samples, width, height, method, crop)


def upscale_to_fit_with_padding(samples: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    _, _, old_height, old_width = samples.shape
    scale = min(width / old_width, height / old_height)
    resized_width = max(1, min(width, round(old_width * scale)))
    resized_height = max(1, min(height, round(old_height * scale)))
    resized = upscale_image(samples, resized_width, resized_height, method, "disabled")
    out = samples.new_zeros((samples.shape[0], samples.shape[1], height, width))
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    out[:, :, y : y + resized_height, x : x + resized_width] = resized
    return out


def upscale_to_kontext_size(samples: torch.Tensor, width: int, height: int, method: str, crop_mode: str) -> torch.Tensor:
    mode = (crop_mode or "crop").strip().lower()
    if mode == "fill":
        return upscale_to_fit_with_padding(samples, width, height, method)
    crop = "disabled" if mode == "disable" else "center"
    return upscale_image(samples, width, height, method, crop)


def reference_image_transform_metadata(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    resize_method: str,
    crop_mode: str,
    sizing_mode: str,
    source_signature: str | None = None,
) -> dict[str, Any]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    mode = (crop_mode or "crop").strip().lower()
    if mode not in {"crop", "disable", "fill"}:
        mode = "crop"

    source_box = (0, 0, int(source_width), int(source_height))
    target_box = (0, 0, int(target_width), int(target_height))

    if mode == "crop":
        old_aspect = source_width / source_height
        new_aspect = target_width / target_height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((source_width - source_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((source_height - source_height * (old_aspect / new_aspect)) / 2)
        source_box = (
            int(x),
            int(y),
            int(source_width - x),
            int(source_height - y),
        )
    elif mode == "fill":
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        x = (target_width - resized_width) // 2
        y = (target_height - resized_height) // 2
        target_box = (int(x), int(y), int(x + resized_width), int(y + resized_height))

    return {
        "source_size": (int(source_width), int(source_height)),
        "target_size": (int(target_width), int(target_height)),
        "source_content_box": source_box,
        "target_content_box": target_box,
        "resize_method": (resize_method or DEFAULT_RESIZE_METHOD).strip().lower(),
        "crop_mode": mode,
        "sizing_mode": sizing_mode,
        "source_signature": source_signature,
    }


def scale_box(
    box: tuple[int, int, int, int],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    from_width, from_height = from_size
    to_width, to_height = to_size
    if from_width == to_width and from_height == to_height:
        return tuple(int(v) for v in box)
    scale_x = to_width / from_width
    scale_y = to_height / from_height
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
        int(round(x2 * scale_x)),
        int(round(y2 * scale_y)),
    )


def clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = tuple(int(v) for v in box)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def pil_resize_filter(method: str | None) -> int:
    method = (method or DEFAULT_RESIZE_METHOD).strip().lower()
    if method in {"lanczos", "nearest-exact"}:
        return _LANCZOS if method == "lanczos" else _NEAREST
    if method == "area":
        return _BOX
    if method == "bilinear":
        return _BILINEAR
    if method == "nearest":
        return _NEAREST
    return _BICUBIC


def flatten_refine_info_items(info: dict[str, Any] | list[Any] | None) -> list[Any]:
    if info is None:
        return []
    values = info if isinstance(info, list) else [info]
    items = []
    for value in values:
        if isinstance(value, list):
            items.extend(flatten_refine_info_items(value))
            continue
        if not isinstance(value, dict):
            continue
        value_items = value.get("items")
        if isinstance(value_items, list):
            items.extend(value_items)
        elif value_items is not None:
            items.append(value_items)
    return items


def update_info_with_kontext_transforms(
    info: dict[str, Any] | list[Any] | None,
    slot_transforms: dict[str, list[dict[str, Any] | None]],
) -> dict[str, Any]:
    base_info = {}
    values = info if isinstance(info, list) else [info]
    for value in values:
        if isinstance(value, dict):
            base_info = value.copy()
            break
    items = flatten_refine_info_items(info)
    if not items or not slot_transforms:
        base_info["items"] = items
        return base_info

    def transform_at(slot: str, index: int) -> dict[str, Any] | None:
        transforms = slot_transforms.get(slot) or []
        if not transforms:
            return None
        return transforms[min(index, len(transforms) - 1)]

    def item_model_size(item: dict[str, Any]) -> tuple[int, int] | None:
        model_image = item.get("model_image")
        if isinstance(model_image, Image.Image):
            return tuple(int(v) for v in model_image.size)
        size = item.get("model_size") or item.get("model_image_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        return None

    def item_model_signature(item: dict[str, Any]) -> str | None:
        model_image = item.get("model_image")
        if isinstance(model_image, Image.Image):
            return image_content_signature(model_image)
        signature = item.get("model_image_signature")
        if isinstance(signature, str):
            return signature
        return None

    def choose_transform(item: dict[str, Any], index: int) -> tuple[dict[str, Any] | None, str]:
        model_signature = item_model_signature(item)
        if model_signature is not None:
            for slot in ("image1", "image2", "image3"):
                selected = transform_at(slot, index)
                if selected and selected.get("source_signature") == model_signature:
                    return selected, slot

        model_size = item_model_size(item)
        if model_size is not None:
            for slot in ("image1", "image2", "image3"):
                selected = transform_at(slot, index)
                if selected and tuple(selected.get("source_size", ())) == model_size:
                    return selected, slot

        for slot in ("image1", "image2", "image3"):
            selected = transform_at(slot, index)
            if selected is not None:
                return selected, slot
        return None, "auto"

    updated_items = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            updated_items.append(item)
            continue
        updated = item.copy()
        selected, selected_source = choose_transform(item, index)
        if selected is not None:
            updated["reference_image_transform"] = selected
            updated["reference_image_transform_source"] = selected_source
        updated_items.append(updated)

    updated_info = base_info.copy()
    updated_info["items"] = updated_items
    return updated_info


def restore_generated_to_model_space(
    generated: Image.Image,
    model_image: Image.Image,
    transform: dict[str, Any] | None,
) -> Image.Image:
    if not isinstance(transform, dict):
        return generated

    mode = (transform.get("crop_mode") or "crop").strip().lower()
    target_size = tuple(transform.get("target_size") or generated.size)
    if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
        target_size = generated.size

    if mode == "fill":
        target_box = tuple(transform.get("target_content_box") or (0, 0, target_size[0], target_size[1]))
        target_box = scale_box(target_box, target_size, generated.size)
        target_box = clamp_box(target_box, generated.size)
        content = generated.crop(target_box)
        return content.resize(model_image.size, _BICUBIC)

    if mode == "crop":
        source_box = tuple(transform.get("source_content_box") or (0, 0, *model_image.size))
        source_box = clamp_box(source_box, model_image.size)
        x1, y1, x2, y2 = source_box
        restored = model_image.copy()
        content = generated.resize((x2 - x1, y2 - y1), _BICUBIC)
        restored.paste(content, (x1, y1))
        return restored

    return generated.resize(model_image.size, _BICUBIC)


def restore_mask_to_model_space(
    mask_l: Image.Image,
    model_size: tuple[int, int],
    transform: dict[str, Any] | None,
) -> Image.Image:
    mask_l = binary_mask(mask_l)
    if not isinstance(transform, dict):
        return mask_l.resize(model_size, _NEAREST)

    mode = (transform.get("crop_mode") or "crop").strip().lower()
    target_size = tuple(transform.get("target_size") or mask_l.size)
    if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
        target_size = mask_l.size

    if mask_l.size != tuple(target_size):
        mask_l = mask_l.resize(tuple(target_size), _NEAREST)

    if mode == "fill":
        target_box = tuple(transform.get("target_content_box") or (0, 0, target_size[0], target_size[1]))
        target_box = clamp_box(target_box, tuple(target_size))
        content = mask_l.crop(target_box)
        return binary_mask(content.resize(model_size, _NEAREST))

    if mode == "crop":
        source_box = tuple(transform.get("source_content_box") or (0, 0, *model_size))
        source_box = clamp_box(source_box, model_size)
        restored = Image.new("L", model_size, 0)
        x1, y1, x2, y2 = source_box
        content = mask_l.resize((x2 - x1, y2 - y1), _NEAREST)
        restored.paste(content, (x1, y1))
        return binary_mask(restored)

    return binary_mask(mask_l.resize(model_size, _NEAREST))


def safe_size_tuple(value: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            width = int(value[0])
            height = int(value[1])
        except (TypeError, ValueError):
            return fallback
        if width > 0 and height > 0:
            return width, height
    return fallback


def transformed_content_box(
    transform: dict[str, Any],
    key: str,
    from_size: tuple[int, int],
    to_size: tuple[int, int],
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    box = tuple(transform.get(key) or fallback)
    if len(box) != 4:
        box = fallback
    box = tuple(int(value) for value in box)
    if from_size != to_size:
        box = scale_box(box, from_size, to_size)
    return clamp_box(box, to_size)


def model_image_to_generated_space(
    model_image: Image.Image,
    generated_size: tuple[int, int],
    transform: dict[str, Any] | None,
) -> Image.Image:
    image = model_image.convert("RGB")
    if generated_size[0] <= 0 or generated_size[1] <= 0:
        raise ValueError("Generated image size must be positive.")
    if not isinstance(transform, dict):
        return image.resize(generated_size, _BICUBIC)

    mode = (transform.get("crop_mode") or "crop").strip().lower()
    target_size = safe_size_tuple(transform.get("target_size"), generated_size)
    source_size = safe_size_tuple(transform.get("source_size"), image.size)
    resample = pil_resize_filter(transform.get("resize_method"))

    if mode == "fill":
        target_box = transformed_content_box(
            transform,
            "target_content_box",
            target_size,
            generated_size,
            (0, 0, target_size[0], target_size[1]),
        )
        canvas = Image.new("RGB", generated_size, (0, 0, 0))
        x1, y1, x2, y2 = target_box
        content = image.resize((x2 - x1, y2 - y1), resample)
        canvas.paste(content, (x1, y1))
        return canvas

    if mode == "crop":
        source_box = transformed_content_box(
            transform,
            "source_content_box",
            source_size,
            image.size,
            (0, 0, source_size[0], source_size[1]),
        )
        return image.crop(source_box).resize(generated_size, resample)

    return image.resize(generated_size, resample)


def model_mask_to_generated_space(
    mask_l: Image.Image,
    generated_size: tuple[int, int],
    transform: dict[str, Any] | None,
) -> Image.Image:
    mask_l = binary_mask(mask_l)
    if generated_size[0] <= 0 or generated_size[1] <= 0:
        raise ValueError("Generated image size must be positive.")
    if not isinstance(transform, dict):
        return binary_mask(mask_l.resize(generated_size, _NEAREST))

    mode = (transform.get("crop_mode") or "crop").strip().lower()
    target_size = safe_size_tuple(transform.get("target_size"), generated_size)
    source_size = safe_size_tuple(transform.get("source_size"), mask_l.size)

    if mode == "fill":
        target_box = transformed_content_box(
            transform,
            "target_content_box",
            target_size,
            generated_size,
            (0, 0, target_size[0], target_size[1]),
        )
        canvas = Image.new("L", generated_size, 0)
        x1, y1, x2, y2 = target_box
        content = mask_l.resize((x2 - x1, y2 - y1), _NEAREST)
        canvas.paste(content, (x1, y1))
        return binary_mask(canvas)

    if mode == "crop":
        source_box = transformed_content_box(
            transform,
            "source_content_box",
            source_size,
            mask_l.size,
            (0, 0, source_size[0], source_size[1]),
        )
        return binary_mask(mask_l.crop(source_box).resize(generated_size, _NEAREST))

    return binary_mask(mask_l.resize(generated_size, _NEAREST))


def model_paste_mask_to_generated_space(
    mask_l: Image.Image,
    generated_size: tuple[int, int],
    transform: dict[str, Any] | None,
) -> Image.Image:
    mask_l = mask_l.convert("L")
    if generated_size[0] <= 0 or generated_size[1] <= 0:
        raise ValueError("Generated image size must be positive.")
    if not isinstance(transform, dict):
        return mask_l.resize(generated_size, _BILINEAR)

    mode = (transform.get("crop_mode") or "crop").strip().lower()
    target_size = safe_size_tuple(transform.get("target_size"), generated_size)
    source_size = safe_size_tuple(transform.get("source_size"), mask_l.size)

    if mode == "fill":
        target_box = transformed_content_box(
            transform,
            "target_content_box",
            target_size,
            generated_size,
            (0, 0, target_size[0], target_size[1]),
        )
        canvas = Image.new("L", generated_size, 0)
        x1, y1, x2, y2 = target_box
        content = mask_l.resize((x2 - x1, y2 - y1), _BILINEAR)
        canvas.paste(content, (x1, y1))
        return canvas

    if mode == "crop":
        source_box = transformed_content_box(
            transform,
            "source_content_box",
            source_size,
            mask_l.size,
            (0, 0, source_size[0], source_size[1]),
        )
        return mask_l.crop(source_box).resize(generated_size, _BILINEAR)

    return mask_l.resize(generated_size, _BILINEAR)


def restore_mask_to_original_space(mask_l: Image.Image, item: dict[str, Any]) -> Image.Image:
    original = item["origin_image"]
    model_image = item["model_image"]
    model_size = model_image.size if isinstance(model_image, Image.Image) else tuple(item.get("model_size") or original.size)
    model_mask = restore_mask_to_model_space(mask_l, model_size, item.get("reference_image_transform"))
    crop_box = item.get("crop_box")
    if crop_box:
        crop_box = tuple(int(value) for value in crop_box)
        restored = Image.new("L", original.size, 0)
        restored.paste(model_mask.resize((crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]), _NEAREST), (crop_box[0], crop_box[1]))
        return binary_mask(restored)
    return binary_mask(model_mask.resize(original.size, _NEAREST))
