from __future__ import annotations

import io
import hashlib
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageCms, ImageFilter


VAE_IMAGE_SIZE = 1024 * 1024
DEFAULT_RESIZE_METHOD = "lanczos"
IMAGE_RESIZE_METHODS = ["lanczos", "area", "bicubic", "bilinear", "nearest-exact"]
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

_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


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


def bbox_from_mask(mask_l: Image.Image) -> tuple[int, int, int, int]:
    arr = np.array(mask_l.convert("L"), dtype=np.uint8)
    ys, xs = np.where(arr > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Mask is empty; paint or connect a non-empty mask.")
    w, h = mask_l.size
    return (
        max(0, int(xs.min())),
        max(0, int(ys.min())),
        min(w, int(xs.max()) + 1),
        min(h, int(ys.max()) + 1),
    )


def bbox_from_mask_or_none(mask_l: Image.Image | None) -> tuple[int, int, int, int] | None:
    if mask_l is None:
        return None
    try:
        return bbox_from_mask(mask_l)
    except ValueError:
        return None


def focus_crop(
    image: Image.Image,
    mask_l: Image.Image,
    bbox: tuple[int, int, int, int],
    margin: int,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    iw, ih = image.size
    if iw <= 0 or ih <= 0:
        return image, mask_l, (0, 0, iw, ih)

    scale = math.sqrt(1024 * 1024 / float(iw * ih))
    x1, y1, x2, y2 = bbox
    m = max(0, int(margin))

    cx1 = max(0, int(math.floor(max(0.0, x1 * scale - m) / scale)))
    cy1 = max(0, int(math.floor(max(0.0, y1 * scale - m) / scale)))
    cx2 = min(iw, int(math.ceil(min(iw * scale, x2 * scale + m) / scale)))
    cy2 = min(ih, int(math.ceil(min(ih * scale, y2 * scale + m) / scale)))

    if cx2 <= cx1:
        cx2 = min(iw, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(ih, cy1 + 1)

    crop_box = (cx1, cy1, cx2, cy2)
    return image.crop(crop_box), mask_l.crop(crop_box), crop_box


def offset_bbox(
    bbox: tuple[int, int, int, int],
    dx: int,
    dy: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 - dx, y1 - dy, x2 - dx, y2 - dy)


def binary_mask(mask_l: Image.Image) -> Image.Image:
    arr = np.where(np.array(mask_l.convert("L"), dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def bbox_mask(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    from PIL import ImageDraw

    w, h = size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), fill=255)
    return mask


def make_spatial_mask(
    mask_l: Image.Image,
    source: str,
    bbox: tuple[int, int, int, int] | None,
) -> Image.Image:
    if (source or "mask").strip().lower() == "bbox" and bbox is not None:
        return bbox_mask(mask_l.size, bbox)
    return binary_mask(mask_l)


def prepare_paste_mask(
    mask_l: Image.Image,
    mask_grow: int,
    blend_blur: int,
) -> Image.Image:
    mask = mask_l.convert("L")
    if mask_grow > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=2 * int(mask_grow) + 1))
    if blend_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(blend_blur)))
    return mask


def composite_masked(
    destination: Image.Image,
    source: Image.Image,
    mask_l: Image.Image,
) -> Image.Image:
    dst = np.asarray(destination.convert("RGB")).astype(np.float32)
    src = np.asarray(source.convert("RGB").resize(destination.size, _BICUBIC)).astype(np.float32)
    alpha = np.asarray(mask_l.convert("L").resize(destination.size, _BILINEAR)).astype(np.float32) / 255.0
    out = src * alpha[:, :, None] + dst * (1.0 - alpha[:, :, None])
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def tensor_image_to_pil(images: torch.Tensor, index: int) -> Image.Image:
    tensor = images.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[min(index, tensor.shape[0] - 1)]
    if tensor.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got {tuple(images.shape)}")
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
    if tensor.shape[-1] != 3:
        raise ValueError(f"Expected IMAGE tensor channel count 3, got {tensor.shape[-1]}")
    arr = (tensor.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def tensor_mask_to_pil(masks: torch.Tensor, index: int, size: tuple[int, int]) -> Image.Image:
    tensor = masks.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[min(index, tensor.shape[0] - 1)]
        if tensor.shape[-1] > 1:
            tensor = tensor.max(dim=-1).values
        else:
            tensor = tensor.squeeze(-1)
    elif tensor.ndim == 3:
        tensor = tensor[min(index, tensor.shape[0] - 1)]
    elif tensor.ndim != 2:
        raise ValueError(f"Expected MASK tensor with 2, 3, or 4 dims, got {tuple(masks.shape)}")

    if tensor.ndim != 2:
        raise ValueError(f"Expected MASK tensor to resolve to 2 dims, got {tuple(tensor.shape)}")

    arr = (tensor.clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if img.size != size:
        img = img.resize(size, _NEAREST)
    return img


def pil_to_tensor_image(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def pil_to_tensor_mask(mask_l: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask_l.convert("L")).astype(np.float32) / 255.0
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


def upscale_image(samples: torch.Tensor, width: int, height: int, method: str, crop: str = "disabled") -> torch.Tensor:
    import comfy.utils

    method = (method or DEFAULT_RESIZE_METHOD).strip().lower()
    if method == "nearest":
        method = "nearest-exact"

    try:
        return comfy.utils.common_upscale(samples, width, height, method, crop)
    except Exception:
        fallback = "area" if method != "area" else "bicubic"
        return comfy.utils.common_upscale(samples, width, height, fallback, crop)


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


def update_info_with_kontext_transforms(
    info: dict[str, Any] | None,
    slot_transforms: dict[str, list[dict[str, Any] | None]],
) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"items": []}
    items = info.get("items")
    if not items or not slot_transforms:
        return info

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

    updated_info = info.copy()
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
    FUNCTION = "preprocess"
    CATEGORY = "RefineNode"

    def preprocess(
        self,
        image: torch.Tensor,
        focus_crop: bool,
        focus_crop_margin: int,
        spatial_prompt_source: str,
        mask: torch.Tensor | None = None,
    ):
        batch = image_batch_size(image)
        model_images = []
        spatial_images = []
        model_masks = []
        infos = []

        for index in range(batch):
            original = normalize_to_srgb(tensor_image_to_pil(image, index))
            if mask is None:
                mask_l = Image.new("L", original.size, 0)
            else:
                mask_l = tensor_mask_to_pil(mask, index, original.size)
            bbox_raw = bbox_from_mask_or_none(mask_l)
            has_region = bbox_raw is not None

            model_image = original
            model_mask = mask_l
            crop_box = None
            bbox_model = bbox_raw

            if focus_crop and bbox_raw is not None:
                model_image, model_mask, crop_box = focus_crop_region(
                    original,
                    mask_l,
                    bbox_raw,
                    int(focus_crop_margin),
                )
                bbox_model = offset_bbox(bbox_raw, crop_box[0], crop_box[1])

            spatial_mask = make_spatial_mask(model_mask, spatial_prompt_source, bbox_model)

            model_images.append(pil_to_tensor_image(model_image))
            spatial_images.append(pil_to_tensor_image(spatial_mask.convert("RGB")))
            model_masks.append(pil_to_tensor_mask(model_mask))
            infos.append(
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
                    "spatial_prompt_source": spatial_prompt_source,
                }
            )

        return (
            torch.stack(model_images, dim=0),
            torch.stack(spatial_images, dim=0),
            torch.stack(model_masks, dim=0),
            {"items": infos},
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
    FUNCTION = "process"
    CATEGORY = "RefineNode"

    def process(
        self,
        image1: torch.Tensor,
        fit_kontext_size: bool,
        resize_method: str,
        crop_mode: str,
        image2: torch.Tensor | None = None,
        image3: torch.Tensor | None = None,
        info: dict[str, Any] | None = None,
    ):
        batch = max(image_batch_size(image1), image_batch_size(image2), image_batch_size(image3))
        image1_outputs = []
        image2_outputs = []
        image3_outputs = []
        slot_transforms: dict[str, list[dict[str, Any] | None]] = {
            "image1": [],
            "image2": [],
            "image3": [],
        }
        target_size = None

        for index in range(batch):
            samples1 = tensor_image_to_pil(image1, index)
            first = normalize_to_srgb(samples1)
            if fit_kontext_size:
                width, height = flux_kontext_target_size(*first.size)
                sizing_mode = "flux_kontext"
            else:
                width, height = calculate_dimensions_area(VAE_IMAGE_SIZE, first.size[0], first.size[1], 8)
                sizing_mode = "area_1024"

            if target_size is None:
                target_size = (width, height)
            elif (width, height) != target_size:
                raise ValueError(
                    "Batch items resolve to different Flux Kontext sizes. "
                    "Split the batch or use images with the same aspect ratio."
                )

            processed = []
            for slot, image in (("image1", image1), ("image2", image2), ("image3", image3)):
                if image is None:
                    slot_transforms[slot].append(None)
                    processed.append(image1.detach().new_zeros((height, width, 3)))
                    continue
                source_pil = normalize_to_srgb(tensor_image_to_pil(image, index))
                slot_transforms[slot].append(
                    reference_image_transform_metadata(
                        source_pil.size,
                        (width, height),
                        resize_method,
                        crop_mode,
                        sizing_mode,
                        image_content_signature(source_pil),
                    )
                )
                sample = image.detach()
                if sample.ndim == 4:
                    sample = sample[min(index, sample.shape[0] - 1)].unsqueeze(0)
                elif sample.ndim == 3:
                    sample = sample.unsqueeze(0)
                else:
                    raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got {tuple(image.shape)}")
                if sample.shape[-1] == 1:
                    sample = sample.repeat(1, 1, 1, 3)
                if sample.shape[-1] != 3:
                    raise ValueError(f"Expected IMAGE tensor channel count 3, got {sample.shape[-1]}")
                sample = sample.movedim(-1, 1)
                out = upscale_to_kontext_size(sample, width, height, resize_method, crop_mode).movedim(1, -1)
                processed.append(out[0])

            image1_outputs.append(processed[0])
            image2_outputs.append(processed[1])
            image3_outputs.append(processed[2])

        if target_size is None:
            raise ValueError("Missing input images.")

        return (
            torch.stack(image1_outputs, dim=0),
            torch.stack(image2_outputs, dim=0),
            torch.stack(image3_outputs, dim=0),
            update_info_with_kontext_transforms(info, slot_transforms),
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
        items = info.get("items") if isinstance(info, dict) else None
        if not items:
            raise ValueError("Missing RefineNode Preprocess Mask info.")

        outputs = []
        masks = []
        batch = max(image_batch_size(generated_image), len(items))
        mode = (paste_back_mode or "mask").strip().lower()

        for index in range(batch):
            item = items[min(index, len(items) - 1)]
            generated = normalize_to_srgb(tensor_image_to_pil(generated_image, index))
            original = item["origin_image"]
            model_image = item["model_image"]
            model_mask = item["model_mask"]
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
                result_crop = generated.resize(model_image.size, _BICUBIC)
            elif mode == "bbox":
                paste_mask = bbox_mask(model_image.size, bbox_model)
                paste_mask = prepare_paste_mask(paste_mask, int(mask_grow), int(blend_blur))
                result_crop = composite_masked(model_image, generated, paste_mask)
            else:
                paste_mask = binary_mask(model_mask)
                paste_mask = prepare_paste_mask(paste_mask, int(mask_grow), int(blend_blur))
                result_crop = composite_masked(model_image, generated, paste_mask)

            if crop_box:
                result = original.copy()
                result.paste(result_crop, (crop_box[0], crop_box[1]))
                full_mask = Image.new("L", original.size, 0)
                full_mask.paste(paste_mask.resize(result_crop.size, _BILINEAR), (crop_box[0], crop_box[1]))
            else:
                result = result_crop
                full_mask = paste_mask

            outputs.append(pil_to_tensor_image(result))
            masks.append(pil_to_tensor_mask(full_mask))

        return (torch.stack(outputs, dim=0), torch.stack(masks, dim=0))


def focus_crop_region(
    image: Image.Image,
    mask_l: Image.Image,
    bbox: tuple[int, int, int, int],
    margin: int,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    return focus_crop(image, mask_l, bbox, margin)


NODE_CLASS_MAPPINGS = {
    "RefineNodePreprocessMask": RefineNodePreprocessMask,
    "RefineNodeReferenceImageProcess": RefineNodeReferenceImageProcess,
    "RefineNodePasteBack": RefineNodePasteBack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RefineNodePreprocessMask": "RefineNode Preprocess Mask",
    "RefineNodeReferenceImageProcess": "RefineNode Reference Image Process",
    "RefineNodePasteBack": "RefineNode Paste Back",
}
