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
        if tensor.shape[-1] == 1 and tensor.shape[0] > 4 and tensor.shape[1] > 4:
            tensor = tensor.squeeze(-1)
        else:
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


def mask_batch_size(masks: torch.Tensor | None) -> int:
    if masks is None:
        return 0
    if masks.ndim == 2:
        return 1
    if masks.ndim == 3 and masks.shape[-1] == 1 and masks.shape[0] > 4 and masks.shape[1] > 4:
        return 1
    if masks.ndim in (3, 4):
        return int(masks.shape[0])
    return 1


def mask_tensor_spatial_size(masks: torch.Tensor) -> tuple[int, int]:
    if masks.ndim == 2:
        return (int(masks.shape[1]), int(masks.shape[0]))
    if masks.ndim == 3 and masks.shape[-1] == 1 and masks.shape[0] > 4 and masks.shape[1] > 4:
        return (int(masks.shape[1]), int(masks.shape[0]))
    if masks.ndim == 4:
        return (int(masks.shape[2]), int(masks.shape[1]))
    return (int(masks.shape[-1]), int(masks.shape[-2]))


def mask_indices_for_image(image_index: int, image_count: int, mask_count: int) -> list[int]:
    if mask_count <= 0:
        return []
    if image_count <= 1:
        return list(range(mask_count))
    if mask_count == 1:
        return [0]
    if mask_count == image_count:
        return [image_index]
    if mask_count % image_count == 0:
        masks_per_image = mask_count // image_count
        start = image_index * masks_per_image
        return list(range(start, start + masks_per_image))
    return [min(image_index, mask_count - 1)]


def combine_masks(masks: torch.Tensor, indices: list[int], size: tuple[int, int]) -> Image.Image:
    if not indices:
        return Image.new("L", size, 0)
    combined = np.zeros((size[1], size[0]), dtype=np.uint8)
    for mask_index in indices:
        mask_l = tensor_mask_to_pil(masks, mask_index, size)
        combined = np.maximum(combined, np.asarray(mask_l.convert("L"), dtype=np.uint8))
    return Image.fromarray(combined, mode="L")


def union_mask_images(mask_images: list[Image.Image], size: tuple[int, int] | None = None) -> Image.Image:
    if not mask_images:
        if size is None:
            raise ValueError("Cannot union an empty mask list without a target size.")
        return Image.new("L", size, 0)
    target_size = size or mask_images[0].size
    combined = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    for mask_l in mask_images:
        current = mask_l.convert("L")
        if current.size != target_size:
            current = current.resize(target_size, _NEAREST)
        combined = np.maximum(combined, np.asarray(current, dtype=np.uint8))
    return Image.fromarray(combined, mode="L")


def connected_component_labels(binary: np.ndarray) -> tuple[np.ndarray, int]:
    binary_u8 = binary.astype(np.uint8, copy=False)
    try:
        import cv2

        label_count, labels = cv2.connectedComponents(binary_u8, connectivity=8)
        return labels.astype(np.int32, copy=False), max(0, int(label_count) - 1)
    except Exception:
        pass

    try:
        from scipy import ndimage

        labels, label_count = ndimage.label(binary_u8, structure=np.ones((3, 3), dtype=np.uint8))
        return labels.astype(np.int32, copy=False), int(label_count)
    except Exception:
        pass

    height, width = binary_u8.shape
    labels = np.zeros((height, width), dtype=np.int32)
    label_count = 0
    ys, xs = np.nonzero(binary_u8)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if labels[start_y, start_x] != 0:
            continue
        label_count += 1
        labels[start_y, start_x] = label_count
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if labels[ny, nx] == 0 and binary_u8[ny, nx]:
                        labels[ny, nx] = label_count
                        stack.append((ny, nx))
    return labels, label_count


def split_mask_components(mask_l: Image.Image) -> list[Image.Image]:
    arr = np.asarray(mask_l.convert("L"), dtype=np.uint8)
    binary = arr > 0
    if not binary.any():
        return []

    labels, label_count = connected_component_labels(binary)
    if label_count <= 1:
        return [mask_l.convert("L")]

    components = []
    for label in range(1, label_count + 1):
        component_pixels = labels == label
        if not component_pixels.any():
            continue
        ys, xs = np.nonzero(component_pixels)
        component = np.zeros_like(arr)
        component[component_pixels] = arr[component_pixels]
        components.append(
            (
                int(ys.min()),
                int(xs.min()),
                Image.fromarray(component, mode="L"),
            )
        )
    components.sort(key=lambda item: (item[0], item[1]))
    return [component for _, _, component in components]


def mask_foreground_area(mask_l: Image.Image) -> int:
    return int((np.asarray(mask_l.convert("L"), dtype=np.uint8) > 0).sum())


def filter_mask_components_by_area_ratio(
    components: list[Image.Image],
    min_area_ratio: float,
    empty_size: tuple[int, int],
) -> list[Image.Image]:
    if not components:
        return [Image.new("L", empty_size, 0)]
    min_area_ratio = clamp_unit_float(min_area_ratio, 0.0)
    if min_area_ratio <= 0.0:
        return components

    areas = [mask_foreground_area(component) for component in components]
    largest_area = max(areas) if areas else 0
    if largest_area <= 0:
        return [Image.new("L", empty_size, 0)]

    kept = [
        component
        for component, area in zip(components, areas)
        if (area / largest_area) >= min_area_ratio
    ]
    if kept:
        return kept

    largest_index = max(range(len(components)), key=lambda index: (areas[index], -index))
    return [components[largest_index]]


def split_mask_images_to_components(mask_images: list[Image.Image], min_area_ratio: float = 0.0) -> list[Image.Image]:
    components = []
    for mask_l in mask_images:
        current_components = split_mask_components(mask_l)
        if current_components:
            components.extend(current_components)
    if components:
        return filter_mask_components_by_area_ratio(components, min_area_ratio, mask_images[0].size)
    if not mask_images:
        raise ValueError("Missing input mask.")
    return [Image.new("L", mask_images[0].size, 0)]


def flatten_mask_input(masks: torch.Tensor | list[torch.Tensor]) -> list[Image.Image]:
    values = masks if isinstance(masks, list) else [masks]
    mask_images = []
    expected_size = None
    for value in values:
        if value is None:
            continue
        count = mask_batch_size(value)
        if count <= 0:
            continue
        size = mask_tensor_spatial_size(value)
        for index in range(count):
            mask_l = tensor_mask_to_pil(value, index, size)
            if expected_size is None:
                expected_size = mask_l.size
            elif mask_l.size != expected_size:
                raise ValueError("All input masks must have the same size.")
            mask_images.append(mask_l)
    if not mask_images:
        raise ValueError("Missing input mask.")
    return mask_images


def stack_mask_images(mask_images: list[Image.Image]) -> torch.Tensor:
    if not mask_images:
        raise ValueError("Cannot output an empty mask batch.")
    size = mask_images[0].size
    normalized = []
    for mask_l in mask_images:
        if mask_l.size != size:
            raise ValueError("All output masks must have the same size.")
        normalized.append(pil_to_tensor_mask(mask_l))
    return torch.stack(normalized, dim=0)


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return max(int(dx), int(dy))


def union_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not bboxes:
        return None
    return (
        min(box[0] for box in bboxes),
        min(box[1] for box in bboxes),
        max(box[2] for box in bboxes),
        max(box[3] for box in bboxes),
    )


def clamp_unit_float(value: float | int | list[float] | list[int], default: float = 1.0) -> float:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def clamp_precision(value: float | int | list[float] | list[int]) -> float:
    return clamp_unit_float(value, 1.0)


def group_entries_by_mst_precision(
    entries: list[tuple[int, Image.Image, tuple[int, int, int, int]]],
    precision: float,
) -> dict[int, list[Image.Image]]:
    parents = list(range(len(entries)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        if root_a < root_b:
            parents[root_b] = root_a
        else:
            parents[root_a] = root_b
        return True

    edges = []
    for left_index in range(len(entries)):
        for right_index in range(left_index + 1, len(entries)):
            gap = bbox_gap(entries[left_index][2], entries[right_index][2])
            edges.append((gap, left_index, right_index))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))

    mst_edges = []
    mst_parents = list(range(len(entries)))

    def mst_find(index: int) -> int:
        while mst_parents[index] != index:
            mst_parents[index] = mst_parents[mst_parents[index]]
            index = mst_parents[index]
        return index

    def mst_merge(a: int, b: int) -> bool:
        root_a = mst_find(a)
        root_b = mst_find(b)
        if root_a == root_b:
            return False
        if root_a < root_b:
            mst_parents[root_b] = root_a
        else:
            mst_parents[root_a] = root_b
        return True

    for gap, left_index, right_index in edges:
        if mst_merge(left_index, right_index):
            mst_edges.append((gap, left_index, right_index))
            if len(mst_edges) == len(entries) - 1:
                break

    remaining_edges = [edge for edge in mst_edges if edge[0] > 0]
    if precision <= 0.0:
        selected_edges = mst_edges
    else:
        extra_count = int(round((1.0 - precision) * len(remaining_edges)))
        selected_edges = remaining_edges[:extra_count]

    for _, left_index, right_index in selected_edges:
        merge(left_index, right_index)

    groups: dict[int, list[Image.Image]] = {}
    first_indices: dict[int, int] = {}
    for entry_index, (original_index, mask_l, _) in enumerate(entries):
        root = find(entry_index)
        groups.setdefault(root, []).append(mask_l)
        first_indices[root] = min(first_indices.get(root, original_index), original_index)
    return {
        root: groups[root]
        for root in sorted(groups, key=lambda value: first_indices[value])
    }


def group_mask_images_by_precision(
    mask_images: list[Image.Image],
    precision: float,
    split_components: bool = True,
    min_area_ratio: float = 0.0,
) -> list[Image.Image]:
    if not mask_images:
        raise ValueError("Missing input mask.")
    if split_components:
        mask_images = split_mask_images_to_components(mask_images, min_area_ratio)
    size = mask_images[0].size
    if precision <= 0.0:
        return [union_mask_images(mask_images, size)]
    entries = []
    for index, mask_l in enumerate(mask_images):
        bbox = bbox_from_mask_or_none(mask_l)
        if bbox is not None:
            entries.append((index, mask_l, bbox))
    if not entries:
        return [Image.new("L", size, 0)]
    if len(entries) == 1:
        return [entries[0][1].convert("L")]
    groups = group_entries_by_mst_precision(entries, precision)
    return [union_mask_images(group, size) for group in groups.values()]


def mask_feature_in_group(
    mask_l: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    group_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    group_width = max(1, group_bbox[2] - group_bbox[0])
    group_height = max(1, group_bbox[3] - group_bbox[1])
    group_area = max(1, group_width * group_height)
    if bbox is None:
        return {
            "bbox": None,
            "center_x": 0.5,
            "center_y": 0.5,
            "x1": 0.5,
            "y1": 0.5,
            "x2": 0.5,
            "y2": 0.5,
            "width": 0.0,
            "height": 0.0,
            "area": 0.0,
            "aspect": 1.0,
        }
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = float((np.asarray(mask_l.convert("L"), dtype=np.uint8) > 0).sum())
    return {
        "bbox": bbox,
        "center_x": (((x1 + x2) / 2.0) - group_bbox[0]) / group_width,
        "center_y": (((y1 + y2) / 2.0) - group_bbox[1]) / group_height,
        "x1": (x1 - group_bbox[0]) / group_width,
        "y1": (y1 - group_bbox[1]) / group_height,
        "x2": (x2 - group_bbox[0]) / group_width,
        "y2": (y2 - group_bbox[1]) / group_height,
        "width": width / group_width,
        "height": height / group_height,
        "area": area / group_area,
        "aspect": width / height,
    }


def mask_match_features(mask_images: list[Image.Image]) -> list[dict[str, Any]]:
    bboxes = [bbox_from_mask_or_none(mask_l) for mask_l in mask_images]
    active_bboxes = [bbox for bbox in bboxes if bbox is not None]
    group_bbox = union_bbox(active_bboxes)
    if group_bbox is None:
        group_bbox = (0, 0, max(1, mask_images[0].size[0]), max(1, mask_images[0].size[1]))
    return [mask_feature_in_group(mask_l, bbox, group_bbox) for mask_l, bbox in zip(mask_images, bboxes)]


def mask_feature_match_score(anchor: dict[str, Any], candidate: dict[str, Any]) -> float:
    if anchor["bbox"] is None and candidate["bbox"] is None:
        return 0.0
    if anchor["bbox"] is None or candidate["bbox"] is None:
        return 1_000_000.0

    center_distance = math.hypot(anchor["center_x"] - candidate["center_x"], anchor["center_y"] - candidate["center_y"])
    area_distance = abs(math.log((candidate["area"] + 1e-6) / (anchor["area"] + 1e-6)))
    aspect_distance = abs(math.log((candidate["aspect"] + 1e-6) / (anchor["aspect"] + 1e-6)))
    size_distance = abs(anchor["width"] - candidate["width"]) + abs(anchor["height"] - candidate["height"])
    y_distance = abs(anchor["center_y"] - candidate["center_y"])
    x_distance = abs(anchor["center_x"] - candidate["center_x"])
    y_span_distance = abs(anchor["y1"] - candidate["y1"]) + abs(anchor["y2"] - candidate["y2"])
    x_span_distance = abs(anchor["x1"] - candidate["x1"]) + abs(anchor["x2"] - candidate["x2"])
    return (
        y_distance * 10.0
        + y_span_distance * 8.0
        + x_distance * 1.5
        + x_span_distance * 0.5
        + center_distance * 0.5
        + area_distance * 0.2
        + aspect_distance * 0.1
        + size_distance * 0.35
    )


def empty_anchor_cost(anchor: dict[str, Any]) -> float:
    if anchor["bbox"] is None:
        return 0.0
    return 0.45 + min(1.75, anchor["area"] * 10.0) + anchor["height"] * 0.75 + anchor["width"] * 0.15


def ordered_mask_feature_indices(features: list[dict[str, Any]], indices: list[int]) -> list[int]:
    if not indices:
        return []

    heights = [features[index]["height"] for index in indices if features[index]["bbox"] is not None]
    row_gap = max(0.008, min(0.035, (float(np.median(heights)) if heights else 0.03) * 0.65))
    rows: list[dict[str, Any]] = []

    for index in sorted(indices, key=lambda value: (features[value]["center_y"], features[value]["center_x"], value)):
        feature = features[index]
        best_row = None
        best_distance = float("inf")
        for row_index, row in enumerate(rows):
            distance = abs(feature["center_y"] - row["center_y"])
            if distance <= row_gap and distance < best_distance:
                best_row = row_index
                best_distance = distance
        if best_row is None:
            rows.append(
                {
                    "indices": [index],
                    "center_y": feature["center_y"],
                }
            )
            continue

        row = rows[best_row]
        row["indices"].append(index)
        row["center_y"] = sum(features[value]["center_y"] for value in row["indices"]) / len(row["indices"])

    ordered = []
    for row in sorted(rows, key=lambda item: item["center_y"]):
        ordered.extend(sorted(row["indices"], key=lambda value: (features[value]["center_x"], features[value]["center_y"], value)))
    return ordered


def match_candidates_to_anchors(anchor_masks: list[Image.Image], candidate_masks: list[Image.Image]) -> list[Image.Image]:
    if not anchor_masks:
        raise ValueError("Missing anchor masks.")
    if not candidate_masks:
        return [Image.new("L", anchor_masks[0].size, 0) for _ in anchor_masks]

    output_size = candidate_masks[0].size
    anchor_features = mask_match_features(anchor_masks)
    candidate_features = mask_match_features(candidate_masks)
    active_anchor_indices = [index for index, feature in enumerate(anchor_features) if feature["bbox"] is not None]
    active_candidate_indices = [index for index, feature in enumerate(candidate_features) if feature["bbox"] is not None]
    if not active_anchor_indices:
        return [Image.new("L", output_size, 0) for _ in anchor_masks]
    if not active_candidate_indices:
        return [Image.new("L", output_size, 0) for _ in anchor_masks]

    assignments: dict[int, list[Image.Image]] = {index: [] for index in range(len(anchor_masks))}
    ordered_anchors = ordered_mask_feature_indices(anchor_features, active_anchor_indices)
    ordered_candidates = ordered_mask_feature_indices(candidate_features, active_candidate_indices)

    candidate_group_bbox = union_bbox(
        [candidate_features[index]["bbox"] for index in ordered_candidates if candidate_features[index]["bbox"] is not None]
    )
    if candidate_group_bbox is None:
        candidate_group_bbox = (0, 0, output_size[0], output_size[1])
    segment_cache: dict[tuple[int, int, int], tuple[float, Image.Image]] = {}

    def segment_cost(anchor_index: int, start: int, end: int) -> tuple[float, Image.Image]:
        key = (anchor_index, start, end)
        if key in segment_cache:
            return segment_cache[key]
        segment_masks = [candidate_masks[ordered_candidates[index]] for index in range(start, end)]
        segment_mask = union_mask_images(segment_masks, output_size)
        segment_bbox = bbox_from_mask_or_none(segment_mask)
        segment_feature = mask_feature_in_group(segment_mask, segment_bbox, candidate_group_bbox)
        merge_penalty = max(0, end - start - 1) * 0.03
        cost = mask_feature_match_score(anchor_features[anchor_index], segment_feature) + merge_penalty
        segment_cache[key] = (cost, segment_mask)
        return cost, segment_mask

    anchor_count = len(ordered_anchors)
    candidate_count = len(ordered_candidates)
    inf = float("inf")
    dp = [[inf] * (candidate_count + 1) for _ in range(anchor_count + 1)]
    prev: list[list[tuple[str, int] | None]] = [[None] * (candidate_count + 1) for _ in range(anchor_count + 1)]
    dp[0][0] = 0.0

    for anchor_pos in range(anchor_count):
        anchor_index = ordered_anchors[anchor_pos]
        for used_candidates in range(candidate_count + 1):
            current = dp[anchor_pos][used_candidates]
            if current == inf:
                continue

            empty_total = current + empty_anchor_cost(anchor_features[anchor_index])
            if empty_total < dp[anchor_pos + 1][used_candidates]:
                dp[anchor_pos + 1][used_candidates] = empty_total
                prev[anchor_pos + 1][used_candidates] = ("empty", used_candidates)

            for end in range(used_candidates + 1, candidate_count + 1):
                cost, _ = segment_cost(anchor_index, used_candidates, end)
                total = current + cost
                if total < dp[anchor_pos + 1][end]:
                    dp[anchor_pos + 1][end] = total
                    prev[anchor_pos + 1][end] = ("segment", used_candidates)

    used_candidates = candidate_count
    for anchor_pos in range(anchor_count, 0, -1):
        action = prev[anchor_pos][used_candidates]
        if action is None:
            break
        kind, start = action
        anchor_index = ordered_anchors[anchor_pos - 1]
        if kind == "segment":
            _, segment_mask = segment_cost(anchor_index, start, used_candidates)
            assignments[anchor_index].append(segment_mask)
        used_candidates = start

    return [
        union_mask_images(assignments[index], output_size)
        if assignments[index]
        else Image.new("L", output_size, 0)
        for index in range(len(anchor_masks))
    ]


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


def tensor_image_as_list_item(image: Image.Image) -> torch.Tensor:
    return pil_to_tensor_image(image).unsqueeze(0)


def tensor_mask_as_list_item(mask_l: Image.Image) -> torch.Tensor:
    return pil_to_tensor_mask(mask_l).unsqueeze(0)


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
                "precision": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_area_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
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
        precision: float | list[float],
        min_area_ratio: float | list[float],
        mask1: torch.Tensor | list[torch.Tensor] | None = None,
        mask2: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        precision = clamp_precision(precision)
        min_area_ratio = clamp_unit_float(min_area_ratio, 0.0)
        if mask1 is None and mask2 is None:
            raise ValueError("Connect at least one mask input.")
        if mask1 is None:
            mask1 = mask2
            mask2 = None
        mask1_images = flatten_mask_input(mask1)

        if mask2 is None:
            grouped_masks = group_mask_images_by_precision(mask1_images, precision, min_area_ratio=min_area_ratio)
            empty_masks = [Image.new("L", grouped_masks[0].size, 0) for _ in grouped_masks]
            return (stack_mask_images(grouped_masks), stack_mask_images(empty_masks))

        mask2_images = flatten_mask_input(mask2)
        anchor_masks = group_mask_images_by_precision(mask1_images, precision, min_area_ratio=min_area_ratio)
        candidate_masks = split_mask_images_to_components(mask2_images, min_area_ratio)

        matched_mask2 = match_candidates_to_anchors(anchor_masks, candidate_masks)
        return (stack_mask_images(anchor_masks), stack_mask_images(matched_mask2))


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
    OUTPUT_IS_LIST = (True, True, True, True)
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
        image_count = image_batch_size(image)
        mask_count = mask_batch_size(mask)
        model_images = []
        spatial_images = []
        model_masks = []
        infos = []

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

            if focus_crop and bbox_raw is not None:
                model_image, model_mask, crop_box = focus_crop_region(
                    original,
                    mask_l,
                    bbox_raw,
                    int(focus_crop_margin),
                )
                bbox_model = offset_bbox(bbox_raw, crop_box[0], crop_box[1])

            spatial_mask = make_spatial_mask(model_mask, spatial_prompt_source, bbox_model)
            group_id = f"source_image_{source_image_index}"

            model_images.append(tensor_image_as_list_item(model_image))
            spatial_images.append(tensor_image_as_list_item(spatial_mask.convert("RGB")))
            model_masks.append(tensor_mask_as_list_item(model_mask))
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
                    "source_image_index": int(source_image_index),
                    "mask_index": None if mask_index is None else int(mask_index),
                    "mask_indices": [int(value) for value in mask_indices],
                    "component_index": None,
                    "component_count": 0,
                    "group_id": group_id,
                    "combined_mask": False,
                }
            )

        for image_index in range(image_count):
            original = normalize_to_srgb(tensor_image_to_pil(image, image_index))
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
                mask_l = tensor_mask_to_pil(mask, mask_index, original.size)
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
            raise ValueError("Missing RefineNode info for mask restore.")

        masks = []
        for mask_input in mask_inputs:
            mask_size = mask_tensor_spatial_size(mask_input)
            for index in range(mask_batch_size(mask_input)):
                masks.append(tensor_mask_to_pil(mask_input, index, mask_size))
        if not masks:
            raise ValueError("Missing mask for RefineNode Restore Mask To Original.")

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
            raise ValueError("Missing valid RefineNode info items for mask restore.")
        return (outputs,)


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
            raise ValueError("Missing RefineNode Preprocess Mask info.")

        generated_images = []
        for generated_input in generated_inputs:
            for index in range(image_batch_size(generated_input)):
                generated_images.append(normalize_to_srgb(tensor_image_to_pil(generated_input, index)))
        if not generated_images:
            raise ValueError("Missing generated images for RefineNode Paste Back.")

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
            raise ValueError("Missing valid RefineNode Preprocess Mask info items.")

        outputs = []
        masks = []
        for group_id in group_order:
            group = groups[group_id]
            result = group["origin_image"].copy()
            combined_mask = Image.new("L", result.size, 0)

            for item_index, item in group["entries"]:
                generated = generated_images[min(item_index, len(generated_images) - 1)]
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
                        np.asarray(full_mask.convert("L").resize(combined_mask.size, _NEAREST), dtype=np.uint8),
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
        elif mode == "bbox":
            paste_mask = bbox_mask(model_image.size, bbox_model)
            paste_mask = prepare_paste_mask(paste_mask, int(mask_grow), int(blend_blur))
        else:
            paste_mask = binary_mask(model_mask)
            paste_mask = prepare_paste_mask(paste_mask, int(mask_grow), int(blend_blur))

        if crop_box:
            result = current_result.copy()
            current_crop = result.crop(crop_box)
            result_crop = composite_masked(current_crop, generated, paste_mask)
            result.paste(result_crop, (crop_box[0], crop_box[1]))
            full_mask = Image.new("L", original.size, 0)
            full_mask.paste(paste_mask.resize(result_crop.size, _BILINEAR), (crop_box[0], crop_box[1]))
            return result, full_mask

        result = composite_masked(current_result, generated, paste_mask)
        full_mask = paste_mask.resize(current_result.size, _BILINEAR)
        return result, full_mask


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
    "RefineNodePreprocessMask": RefineNodePreprocessMask,
    "RefineNodeReferenceImageProcess": RefineNodeReferenceImageProcess,
    "RefineNodeRestoreMaskToOriginal": RefineNodeRestoreMaskToOriginal,
    "RefineNodePasteBack": RefineNodePasteBack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RefineNodeMaskBatchProcess": "RefineNode Mask Batch Process",
    "RefineNodeSliceAndMatchMasks": "RefineNode Slice And Match Masks",
    "RefineNodePreprocessMask": "RefineNode Preprocess Mask",
    "RefineNodeReferenceImageProcess": "RefineNode Reference Image Process",
    "RefineNodeRestoreMaskToOriginal": "RefineNode Restore Mask To Original",
    "RefineNodePasteBack": "RefineNode Paste Back",
}
