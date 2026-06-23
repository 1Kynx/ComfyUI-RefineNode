from __future__ import annotations

import math
from typing import Any, NoReturn, TypedDict

import numpy as np
import torch
from PIL import Image, ImageFilter


_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")


class RefineEntryRequired(TypedDict):
    origin_image: Image.Image
    origin_size: tuple[int, int]
    model_image: Image.Image
    model_mask: Image.Image
    has_region: bool
    spatial_prompt_source: str
    source_image_index: int
    mask_index: int | None
    mask_indices: list[int]
    group_id: str


class RefineEntry(RefineEntryRequired, total=False):
    spatial_mask: Image.Image
    bbox_raw: tuple[int, int, int, int] | None
    bbox_model: tuple[int, int, int, int] | None
    crop_box: tuple[int, int, int, int] | None
    component_index: int | None
    component_count: int
    combined_mask: bool
    reference_image_transform: dict[str, Any] | None


def refine_error(
    node_name: str,
    msg: str,
    item_index: int | None = None,
    group_id: str | None = None,
) -> NoReturn:
    parts = [f"[{node_name}]"]
    if item_index is not None and item_index >= 0:
        parts.append(f"item={item_index}")
    if group_id:
        parts.append(f"group={group_id}")
    parts.append(msg)
    raise ValueError(" ".join(parts))


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
    margin_orig = max(0, int(margin)) / max(scale, 1e-6)

    cx1 = max(0, int(math.floor(float(x1) - margin_orig)))
    cy1 = max(0, int(math.floor(float(y1) - margin_orig)))
    cx2 = min(iw, int(math.ceil(float(x2) + margin_orig)))
    cy2 = min(ih, int(math.ceil(float(y2) + margin_orig)))

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


def mask_batch_size(masks: torch.Tensor | None) -> int:
    if masks is None:
        return 0
    if masks.ndim == 2:
        return 1
    if masks.ndim in (3, 4):
        return int(masks.shape[0])
    raise ValueError(f"Expected MASK tensor with 2, 3, or 4 dims, got {tuple(masks.shape)}")


def mask_tensor_spatial_size(masks: torch.Tensor) -> tuple[int, int]:
    if masks.ndim == 2:
        return (int(masks.shape[1]), int(masks.shape[0]))
    if masks.ndim == 3:
        return (int(masks.shape[2]), int(masks.shape[1]))
    if masks.ndim == 4:
        return (int(masks.shape[2]), int(masks.shape[1]))
    raise ValueError(f"Expected MASK tensor with 2, 3, or 4 dims, got {tuple(masks.shape)}")


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
    refine_error(
        "RefineNode Preprocess Mask",
        f"received an incompatible image/mask batch: {image_count} images and {mask_count} masks. "
        "Use one mask, one mask per image, or a mask count that is evenly divisible by the image count.",
    )


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


def filtered_union_mask(mask_images: list[Image.Image], min_area_ratio: float = 0.0) -> Image.Image:
    components = split_mask_images_to_components(mask_images, min_area_ratio)
    return binary_mask(union_mask_images(components, mask_images[0].size))


def clamp_int_value(value: int | float | list[int] | list[float], default: int, min_value: int, max_value: int) -> int:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))


def normalize_choice(value: str | list[str], default: str, choices: set[str]) -> str:
    if isinstance(value, list):
        value = value[0] if value else default
    text = str(value or default).strip().lower()
    return text if text in choices else default


def normalized_grid_cells(rows: int, columns: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        (
            (row / rows, (row + 1) / rows),
            (column / columns, (column + 1) / columns),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def bbox_grid_cell_box(
    bbox: tuple[int, int, int, int],
    row_interval: tuple[float, float],
    column_interval: tuple[float, float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    row_start, row_end = row_interval
    column_start, column_end = column_interval
    sx = x1 + int(math.floor(width * column_start))
    ex = x1 + int(math.ceil(width * column_end))
    sy = y1 + int(math.floor(height * row_start))
    ey = y1 + int(math.ceil(height * row_end))
    return (
        sx,
        sy,
        max(sx + 1, min(x2, ex)),
        max(sy + 1, min(y2, ey)),
    )


def intersect_mask_with_bbox(mask_l: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    stripe = bbox_mask(mask_l.size, bbox)
    mask_arr = np.asarray(binary_mask(mask_l), dtype=np.uint8)
    stripe_arr = np.asarray(stripe, dtype=np.uint8)
    return Image.fromarray(np.minimum(mask_arr, stripe_arr), mode="L")


def bbox_grid_sliced_masks(
    mask_l: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    cells: list[tuple[tuple[float, float], tuple[float, float]]],
    output_mode: str,
) -> list[Image.Image]:
    if bbox is None:
        return [Image.new("L", mask_l.size, 0) for _ in cells]

    output = []
    for row_interval, column_interval in cells:
        slice_box = bbox_grid_cell_box(bbox, row_interval, column_interval)
        if output_mode == "bbox":
            output.append(bbox_mask(mask_l.size, slice_box))
        else:
            output.append(intersect_mask_with_bbox(mask_l, slice_box))
    return output


def product_mask_descriptors(mask_images: list[Image.Image], min_area_ratio: float) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for index, mask_l in enumerate(mask_images):
        filtered = filtered_union_mask([mask_l], min_area_ratio)
        bbox = bbox_from_mask_or_none(filtered)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        descriptors.append(
            {
                "index": index,
                "mask": filtered,
                "bbox": bbox,
                "center_x": (x1 + x2) * 0.5,
                "center_y": (y1 + y2) * 0.5,
                "area": max(1, mask_foreground_area(filtered)),
                "aspect": max(1e-6, (x2 - x1) / max(1, y2 - y1)),
            }
        )

    if not descriptors:
        return []

    union_bbox = (
        min(desc["bbox"][0] for desc in descriptors),
        min(desc["bbox"][1] for desc in descriptors),
        max(desc["bbox"][2] for desc in descriptors),
        max(desc["bbox"][3] for desc in descriptors),
    )
    ux1, uy1, ux2, uy2 = union_bbox
    union_width = max(1, ux2 - ux1)
    union_height = max(1, uy2 - uy1)
    union_area = max(1, union_width * union_height)
    for desc in descriptors:
        x1, y1, x2, y2 = desc["bbox"]
        desc["norm_x"] = (desc["center_x"] - ux1) / union_width
        desc["norm_y"] = (desc["center_y"] - uy1) / union_height
        desc["norm_left"] = (x1 - ux1) / union_width
        desc["norm_top"] = (y1 - uy1) / union_height
        desc["norm_right"] = (x2 - ux1) / union_width
        desc["norm_bottom"] = (y2 - uy1) / union_height
        desc["norm_area"] = desc["area"] / union_area
    return descriptors


def spatially_order_product_descriptors(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for desc in sorted(descriptors, key=lambda item: (item["norm_y"], item["norm_x"], item["index"])):
        placed = False
        for row in rows:
            overlap = min(desc["norm_bottom"], row["bottom"]) - max(desc["norm_top"], row["top"])
            row_height = max(1e-6, row["bottom"] - row["top"])
            desc_height = max(1e-6, desc["norm_bottom"] - desc["norm_top"])
            center_delta = abs(desc["norm_y"] - row["center"])
            if overlap >= -0.03 or center_delta <= max(0.08, 0.5 * max(row_height, desc_height)):
                row["items"].append(desc)
                row["top"] = min(row["top"], desc["norm_top"])
                row["bottom"] = max(row["bottom"], desc["norm_bottom"])
                row["center"] = sum(item["norm_y"] for item in row["items"]) / len(row["items"])
                placed = True
                break
        if not placed:
            rows.append(
                {
                    "top": desc["norm_top"],
                    "bottom": desc["norm_bottom"],
                    "center": desc["norm_y"],
                    "items": [desc],
                }
            )

    ordered: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["center"], item["top"])):
        ordered.extend(sorted(row["items"], key=lambda item: (item["norm_x"], item["norm_y"], item["index"])))
    return ordered


def product_position_match_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    center_distance = math.hypot(left["norm_x"] - right["norm_x"], left["norm_y"] - right["norm_y"])
    area_delta = abs(math.log(max(left["norm_area"], 1e-6) / max(right["norm_area"], 1e-6)))
    aspect_delta = abs(math.log(max(left["aspect"], 1e-6) / max(right["aspect"], 1e-6)))
    return center_distance * 10.0 + area_delta * 0.75 + aspect_delta * 0.25


def greedy_product_position_assignment(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = [
        (product_position_match_score(left_item, right_item), left_index, right_index)
        for left_index, left_item in enumerate(left)
        for right_index, right_item in enumerate(right)
    ]
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pairs.append((left[left_index], right[right_index]))
        if len(pairs) == min(len(left), len(right)):
            break
    left_order = {id(item): index for index, item in enumerate(left)}
    pairs.sort(key=lambda pair: left_order[id(pair[0])])
    return pairs


def minimum_product_position_assignment(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(left) != len(right):
        return greedy_product_position_assignment(left, right)
    count = len(left)
    if count > 16:
        return greedy_product_position_assignment(left, right)

    memo: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

    def solve(left_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        key = (left_index, used_mask)
        if key in memo:
            return memo[key]
        if left_index >= count:
            return 0.0, ()

        best_cost = float("inf")
        best_order: tuple[int, ...] = ()
        for right_index in range(count):
            bit = 1 << right_index
            if used_mask & bit:
                continue
            rest_cost, rest_order = solve(left_index + 1, used_mask | bit)
            total_cost = product_position_match_score(left[left_index], right[right_index]) + rest_cost
            order = (right_index,) + rest_order
            if total_cost < best_cost - 1e-9 or (abs(total_cost - best_cost) <= 1e-9 and order < best_order):
                best_cost = total_cost
                best_order = order
        memo[key] = best_cost, best_order
        return memo[key]

    _, right_order = solve(0, 0)
    return [(left[index], right[right_index]) for index, right_index in enumerate(right_order)]


def nearest_product_descriptor(
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda item: (product_position_match_score(source, item), item["norm_y"], item["norm_x"], item["index"]),
    )


def position_matched_product_pairs(
    mask1_descriptors: list[dict[str, Any]],
    mask2_descriptors: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ordered_mask1 = spatially_order_product_descriptors(mask1_descriptors)
    ordered_mask2 = spatially_order_product_descriptors(mask2_descriptors)
    if len(ordered_mask1) == len(ordered_mask2):
        return minimum_product_position_assignment(ordered_mask1, ordered_mask2)
    if len(ordered_mask1) > len(ordered_mask2):
        return [(mask1_item, nearest_product_descriptor(mask1_item, ordered_mask2)) for mask1_item in ordered_mask1]

    pairs = [
        (nearest_product_descriptor(mask2_item, ordered_mask1), mask2_item)
        for mask2_item in ordered_mask2
    ]
    mask1_order = {id(item): index for index, item in enumerate(ordered_mask1)}
    pairs.sort(key=lambda pair: (mask1_order[id(pair[0])], pair[1]["norm_y"], pair[1]["norm_x"], pair[1]["index"]))
    return pairs


def slice_mask_pair_by_product_bbox(
    mask1_images: list[Image.Image],
    mask2_images: list[Image.Image] | None,
    min_area_ratio: float,
    rows: int,
    columns: int,
    auto_match_orientation: bool,
    output_mode: str,
) -> tuple[list[Image.Image], list[Image.Image]]:
    mask1_union = filtered_union_mask(mask1_images, min_area_ratio)
    mask1_bbox = bbox_from_mask_or_none(mask1_union)
    mask1_size = mask1_union.size
    mask2_size = mask2_images[0].size if mask2_images else mask1_size

    if mask1_bbox is None:
        return ([Image.new("L", mask1_size, 0)], [Image.new("L", mask2_size, 0)])

    if auto_match_orientation and rows != columns:
        long_count = max(rows, columns)
        short_count = min(rows, columns)
        if (mask1_bbox[2] - mask1_bbox[0]) > (mask1_bbox[3] - mask1_bbox[1]):
            rows, columns = short_count, long_count
        else:
            rows, columns = long_count, short_count

    cells = normalized_grid_cells(rows, columns)

    mask1_slices = bbox_grid_sliced_masks(mask1_union, mask1_bbox, cells, output_mode)
    keep_indices = list(range(len(mask1_slices)))
    if output_mode == "mask":
        non_empty_indices = [
            index
            for index, slice_mask in enumerate(mask1_slices)
            if bbox_from_mask_or_none(slice_mask) is not None
        ]
        keep_indices = non_empty_indices or [0]

    kept_mask1 = [mask1_slices[index] for index in keep_indices]

    if not mask2_images:
        return (kept_mask1, [Image.new("L", mask1_size, 0) for _ in kept_mask1])

    mask2_union = filtered_union_mask(mask2_images, min_area_ratio)
    mask2_bbox = bbox_from_mask_or_none(mask2_union)
    mask2_slices = bbox_grid_sliced_masks(mask2_union, mask2_bbox, cells, output_mode)
    kept_mask2 = [mask2_slices[index] for index in keep_indices]
    return (kept_mask1, kept_mask2)


def slice_masks_by_product_bbox(
    mask1_images: list[Image.Image],
    mask2_images: list[Image.Image] | None,
    min_area_ratio: float,
    rows: int,
    columns: int,
    auto_match_orientation: bool,
    match_mode: str,
    output_mode: str,
) -> tuple[list[Image.Image], list[Image.Image]]:
    if match_mode == "repeat_mask1" and mask2_images:
        output_mask1: list[Image.Image] = []
        output_mask2: list[Image.Image] = []
        for mask2_image in mask2_images:
            pair_mask1, pair_mask2 = slice_mask_pair_by_product_bbox(
                mask1_images,
                [mask2_image],
                min_area_ratio,
                rows,
                columns,
                auto_match_orientation,
                output_mode,
            )
            output_mask1.extend(pair_mask1)
            output_mask2.extend(pair_mask2)
        return output_mask1, output_mask2

    if match_mode == "pair_by_position":
        output_mask1: list[Image.Image] = []
        output_mask2: list[Image.Image] = []
        mask1_descriptors = product_mask_descriptors(mask1_images, min_area_ratio)
        mask1_size = mask1_images[0].size if mask1_images else (1, 1)

        if not mask1_descriptors:
            mask2_size = mask2_images[0].size if mask2_images else mask1_size
            return ([Image.new("L", mask1_size, 0)], [Image.new("L", mask2_size, 0)])

        if not mask2_images:
            for mask1_desc in spatially_order_product_descriptors(mask1_descriptors):
                pair_mask1, pair_mask2 = slice_mask_pair_by_product_bbox(
                    [mask1_desc["mask"]],
                    None,
                    min_area_ratio,
                    rows,
                    columns,
                    auto_match_orientation,
                    output_mode,
                )
                output_mask1.extend(pair_mask1)
                output_mask2.extend(pair_mask2)
            return output_mask1, output_mask2

        mask2_descriptors = product_mask_descriptors(mask2_images, min_area_ratio)
        if not mask2_descriptors:
            mask2_size = mask2_images[0].size
            for mask1_desc in spatially_order_product_descriptors(mask1_descriptors):
                pair_mask1, pair_mask2 = slice_mask_pair_by_product_bbox(
                    [mask1_desc["mask"]],
                    [Image.new("L", mask2_size, 0)],
                    min_area_ratio,
                    rows,
                    columns,
                    auto_match_orientation,
                    output_mode,
                )
                output_mask1.extend(pair_mask1)
                output_mask2.extend(pair_mask2)
            return output_mask1, output_mask2

        for mask1_desc, mask2_desc in position_matched_product_pairs(mask1_descriptors, mask2_descriptors):
            pair_mask1, pair_mask2 = slice_mask_pair_by_product_bbox(
                [mask1_desc["mask"]],
                [mask2_desc["mask"]],
                min_area_ratio,
                rows,
                columns,
                auto_match_orientation,
                output_mode,
            )
            output_mask1.extend(pair_mask1)
            output_mask2.extend(pair_mask2)
        return output_mask1, output_mask2

    if match_mode == "pair_by_index":
        if not mask2_images:
            output_mask1: list[Image.Image] = []
            output_mask2: list[Image.Image] = []
            for mask1_image in mask1_images:
                pair_mask1, pair_mask2 = slice_mask_pair_by_product_bbox(
                    [mask1_image],
                    None,
                    min_area_ratio,
                    rows,
                    columns,
                    auto_match_orientation,
                    output_mode,
                )
                output_mask1.extend(pair_mask1)
                output_mask2.extend(pair_mask2)
            return output_mask1, output_mask2

        output_mask1 = []
        output_mask2 = []
        count = max(len(mask1_images), len(mask2_images))
        for index in range(count):
            mask1_image = mask1_images[min(index, len(mask1_images) - 1)]
            mask2_image = mask2_images[min(index, len(mask2_images) - 1)]
            pair_mask1, pair_mask2 = slice_mask_pair_by_product_bbox(
                [mask1_image],
                [mask2_image],
                min_area_ratio,
                rows,
                columns,
                auto_match_orientation,
                output_mode,
            )
            output_mask1.extend(pair_mask1)
            output_mask2.extend(pair_mask2)
        return output_mask1, output_mask2

    return slice_mask_pair_by_product_bbox(
        mask1_images,
        mask2_images,
        min_area_ratio,
        rows,
        columns,
        auto_match_orientation,
        output_mode,
    )


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


def clamp_unit_float(value: float | int | list[float] | list[int], default: float = 1.0) -> float:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def pil_to_tensor_mask(mask_l: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask_l.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def tensor_mask_as_list_item(mask_l: Image.Image) -> torch.Tensor:
    return pil_to_tensor_mask(mask_l).unsqueeze(0)
