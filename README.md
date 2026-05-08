# ComfyUI-RefineNode

ComfyUI-RefineNode provides a small set of model-agnostic ComfyUI nodes for local repainting, detail repair, product/logo/text refinement, and paste-back compositing.

[RefineAnything](https://github.com/limuloo/RefineAnything) targets region-specific image refinement: given an input image and a user-specified region (e.g., scribble mask or bounding box), it restores fine-grained details--text, logos, thin structures--while keeping all non-edited pixels unchanged. It supports both reference-based and reference-free refinement.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1Kynx/ComfyUI-RefineNode.git
```

Restart ComfyUI after installation.

## Example Workflows

The included example workflows use the Qwen Image Edit model together with the RefineAnything LoRA.

RefineAnything LoRA:
[limuloo1999/RefineAnything](https://huggingface.co/limuloo1999/RefineAnything/tree/main)

You can try different options in the `Edit Model Reference Method` node, but it is best to avoid `index_timestep_zero` because it can introduce a noticeable color shift.

`example_workflows/Reference-based Logo Refinement.json`

- Reference-based workflow for refining logos, text, or product details with a clean reference image.
- In reference-based mode, connect the target refinement image to `image1`, the reference image to `image2`, and the target refinement mask/spatial mask image to `image3`.

![Reference-based Logo Refinement workflow](example_workflows/Reference-based%20Logo%20Refinement.jpeg)

`example_workflows/Reference-free Text Refinement.json`

- Reference-free workflow for refining local text details from the target image and mask only.

![Reference-free Text Refinement workflow](example_workflows/Reference-free%20Text%20Refinement.jpeg)

## Node Details

### RefineNode Mask Batch Process

Prepares a `MASK` tensor before it is sent into `RefineNode Preprocess Mask`.

Inputs:

- `mask`: Input mask. This can be a normal mask, a batched mask, or a ComfyUI mask list.
- `combined_mask`: When disabled, each disconnected painted island is split into a separate mask in the output batch. When enabled, all incoming masks are unioned into one mask.

Outputs:

- `mask`: A processed `MASK` batch.

Use this node when a single `Load Image` mask contains multiple separated painted regions and you want either one refinement job per region or one combined refinement job.

### RefineNode Slice And Match Masks

Slices a single mask set or aligns two mask groups that describe the same product across different views, sizes, or proportions. The node uses the whole filtered product mask bbox as the anchor, then applies the same normalized row/column bbox grid to the second mask group. Each group's original image size is preserved.

Inputs:

- `mask1`: Optional first mask group. Output order is anchored to this group when both inputs are connected.
- `mask2`: Optional second mask group to match against `mask1`.
- `output_mode`: `mask` keeps the original product mask shape inside each bbox slice. `bbox` outputs full rectangular bbox slices for the most stable spatial correspondence. Default is `bbox`.
- `min_area_ratio`: Removes tiny disconnected components before grouping and matching, with `0.01` step precision. Default is `0.01`; `0.0` disables filtering. Values are relative to the largest disconnected component in the same input group, so `0.02` removes components smaller than 2% of that largest component's foreground area.
- `rows`: Number of equal-height bbox grid rows. Default is `4`.
- `columns`: Number of equal-width bbox grid columns. Default is `1`.
- `auto_match_orientation`: Automatically makes the longer split direction follow the longer side of the `mask1` product bbox. Default is enabled.

Outputs:

- `mask1`: First sliced/grouped mask batch.
- `mask2`: Second mask batch aligned to `mask1`. When only one input is connected, this is an empty placeholder batch.

The node splits disconnected painted islands internally, removes components below `min_area_ratio`, unions the remaining regions into one product mask, and slices that product bbox into a row-major grid: top to bottom, left to right. When `auto_match_orientation` is enabled, the larger of `rows` and `columns` follows the longer side of the product bbox: `rows=4, columns=1` stays 4x1 for vertical products and becomes 1x4 for horizontal products, while `rows=1, columns=3` becomes 3x1 for vertical products and stays 1x3 for horizontal products. In `mask` mode, empty `mask1` cells are skipped and `mask2` keeps the same remaining cell positions. In `bbox` mode, the full `rows × columns` rectangular grid is preserved. Use multiple rows and columns for two-dimensional local regions.

### RefineNode Match Product Angle

Rotates one product image to match the main mask angle of a reference product mask. The node rotates the whole source image without scaling, rotates the source mask with the same angle, and crops the result to the rotated source mask bbox.

Inputs:

- `image`: Source product image to rotate.
- `source_mask`: Product mask for the source image.
- `reference_mask`: Product mask whose angle should be matched.
- `canvas_expand`: Expands the output crop from the rotated product bbox toward the full rotated canvas. `0.0` keeps the tight product bbox crop; `1.0` keeps the full rotated canvas.

Outputs:

- `image`: Rotated RGB image cropped to the rotated source mask bbox.
- `mask`: Rotated source mask cropped to the same bbox.
- `angle`: Applied rotation angle in degrees.

The angle is estimated from each mask's foreground principal axis. Empty source or reference masks return the original source image, the source mask, and `angle=0`. The node does not scale, perform perspective correction, or distinguish 0 degrees from 180 degrees.

### RefineNode Rotate Image

Rotates an image by a manually specified angle.

Inputs:

- `image`: Image to rotate.
- `angle`: Rotation angle in degrees.

Outputs:

- `image`: Rotated RGB image.

The image is rotated without scaling. The output canvas expands to contain the full rotated image, and newly exposed pixels are filled with black.

### RefineNode Preprocess Mask

Prepares the target image and optional mask before model inference, and creates the `info` data required for accurate paste-back.

Inputs:

- `image`: The image to refine or use as a reference.
- `mask`: Optional mask. If disconnected, the node uses an empty mask and does not raise an error.
- `focus_crop`: When enabled, crops around the masked region to reduce unrelated image area before generation.
- `focus_crop_margin`: Extra context retained around the mask crop.
- `spatial_prompt_source`: Controls the `spatial_mask_image` output. `mask` keeps the original mask shape, while `bbox` uses the mask bounding box.

Outputs:

- `image`: Processed model input image.
- `spatial_mask_image`: Spatial mask image for image-editing nodes such as Qwen Image Edit Plus.
- `mask`: Mask aligned with the output `image`.
- `info`: Metadata containing the original image, crop position, mask, and paste-back coordinates. Pass this to downstream paste-back nodes.

Mask behavior:

- `RefineNode Preprocess Mask` no longer decides whether masks should be split or combined.
- A batched `MASK` input is treated as the exact refinement job list: one output image-list item per incoming mask, using the existing image/mask batch assignment rules.
- Use `RefineNode Slice And Match Masks` before this node when you need to split painted islands, combine all masks, group nearby mask regions, or align two mask groups.

### RefineNode Reference Image Process

Resizes and aligns the target image, reference image, and mask image to exactly the same size. This is useful for multi-image reference workflows and Qwen Image Edit Plus style inputs.

Inputs:

- `image1`: Required. Usually the target refinement image.
- `image2`: Optional. Usually the reference image.
- `image3`: Optional. In reference-based mode, usually the target refinement mask or `spatial_mask_image`.
- `info`: Optional `REFINENODE_INFO` from `RefineNode Preprocess Mask`. When connected, the node records its resize, crop, or fill transform so `RefineNode Paste Back` can restore the generated image to the correct original position.
- `fit_kontext_size`: When enabled, uses the ComfyUI/Flux Kontext-style target size. When disabled, uses an approximately `1024 x 1024` target area rounded to multiples of 8.
- `resize_method`: Shared resize method for all connected images. The default is `lanczos`.
- `crop_mode`: `crop` center-crops to the target aspect ratio, `disable` resizes directly, and `fill` preserves aspect ratio then pads with black to the target size.

Outputs:

- `image1`: Processed target refinement image.
- `image2`: Processed reference image. If the input is disconnected, this output is a blank placeholder image.
- `image3`: Processed mask image or third reference image. If the input is disconnected, this output is a blank placeholder image.
- `info`: Updated paste-back metadata.

When `image1` or `image3` comes from a multi-mask `RefineNode Preprocess Mask` image list, ComfyUI processes each list item sequentially. A single connected reference image in `image2` is reused for all list items.

Reference-based mode reminder:

- Connect the target refinement image to `image1`.
- Connect the reference image to `image2`.
- Connect the target refinement mask/spatial mask image to `image3`.

### RefineNode Restore Mask To Original

Restores a mask detected on a cropped or resized RefineNode image back to the original image coordinate space.

Inputs:

- `mask`: Mask detected after `RefineNode Preprocess Mask` and, optionally, `RefineNode Reference Image Process`.
- `info`: `REFINENODE_INFO` from `RefineNode Preprocess Mask` or the updated `info` from `RefineNode Reference Image Process`.

Outputs:

- `mask`: Restored mask in the original image size and position.

Use this when you first crop a product with `RefineNode Preprocess Mask`, run another mask detector on the cropped/model image, and then need that newly detected mask to line up with the original uncropped image. If the image also passed through `RefineNode Reference Image Process`, the node reverses that resize/crop/fill transform before restoring the mask through the original `crop_box`.

When the connected mask input contains multiple masks, such as the output from `RefineNode Slice And Match Masks`, every mask is restored. If there is only one `info` item, that same coordinate transform is reused for all mask outputs.

### RefineNode Paste Back

Composites the generated result back into the original image and returns both the final image and the actual paste mask.

Inputs:

- `generated_image`: Image generated by the model.
- `info`: Paste-back metadata from `RefineNode Preprocess Mask` or `RefineNode Reference Image Process`.
- `paste_back_mode`: `mask` pastes using the original mask, while `bbox` pastes using the mask bounding box.
- `mask_grow`: Expands the paste mask before compositing. Useful for reducing tiny uncovered edges.
- `blend_blur`: Blurs and feathers the paste mask to soften seams.

Outputs:

- `image`: Final pasted-back image.
- `paste_mask`: Actual mask used for compositing, useful for previewing and debugging the paste area.

When `generated_image` and `info` arrive as ComfyUI lists, all generated refinement results that belong to the same original source image are pasted back sequentially into that original image. The node returns one final image and one combined paste mask per source image.

## Citation

If you use this repository, please cite:

```bibtex
@article{zhou2026refineanything,
  title={RefineAnything: Multimodal Region-Specific Refinement for Perfect Local Details},
  author={Zhou, Dewei and Li, You and Yang, Zongxin and Yang, Yi},
  journal={arXiv preprint arXiv:2604.06870},
  year={2026}
}
```
