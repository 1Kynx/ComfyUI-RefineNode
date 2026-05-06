# ComfyUI-RefineNode

Generic ComfyUI nodes for local image repainting, detail repair, product/logo/text refinement, and paste-back compositing.

These nodes are model-agnostic. They do not load checkpoints, patch model internals, or require a Diffusers pipeline. Use them before and after any image editing, inpainting, Kontext, Qwen Image Edit Plus, Flux, SD inpaint, or custom repair workflow where you need:

- crop the masked region while preserving paste-back metadata;
- prepare target/reference/mask images at matching sizes;
- send the processed image into any model-specific generation path;
- paste the generated result back into the original image through the mask or bbox.

## Install

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI-RefineNode
```

Restart ComfyUI.

`requirements.txt` intentionally has no extra runtime dependency. The nodes use ComfyUI, PyTorch, PIL, and NumPy from the existing ComfyUI environment.

## Nodes

| Node | Output | Purpose |
|------|--------|---------|
| `RefineNode Preprocess Mask` | `image`, `spatial_mask_image`, `mask`, `info` | Reads an image plus optional mask, optionally focus-crops the mask region, and stores paste-back metadata in `REFINENODE_INFO`. |
| `RefineNode Reference Image Process` | `image1`, `image2`, `image3`, `info` | Resizes one to three images to the same target size with shared resize/crop/fill settings, and updates `REFINENODE_INFO` so paste-back can reverse that transform. |
| `RefineNode Paste Back` | `image`, `paste_mask` | Composites a generated image back into the original image using the original mask or bbox. |

## Generic Wiring

```text
source image + optional mask
  -> RefineNode Preprocess Mask

RefineNode Preprocess Mask.image
  -> your model, VAEEncode, image encoder, inpaint image input, or reference pipeline

RefineNode Preprocess Mask.mask
  -> your model's mask input, if it has one

RefineNode Preprocess Mask.spatial_mask_image
  -> optional visual/spatial condition image

model generated image + RefineNode Preprocess Mask.info
  -> RefineNode Paste Back
```

Use `RefineNode Reference Image Process` when multiple images must share the same pixels before generation:

```text
processed target image
  -> RefineNode Reference Image Process.image1

optional reference image
  -> RefineNode Reference Image Process.image2

optional spatial mask/control image
  -> RefineNode Reference Image Process.image3

matching Preprocess Mask.info
  -> RefineNode Reference Image Process.info

generated image + RefineNode Reference Image Process.info
  -> RefineNode Paste Back
```

The `info` input can come from whichever image you intend to paste back into. The node automatically matches `info.items[*].model_image` against image1/image2/image3 by content signature first, then by size, and records the matched slot's resize/crop/fill transform for paste-back.

## Focus Crop

`RefineNode Preprocess Mask` can crop around the painted mask before generation:

- `focus_crop=true` crops the image and mask around the mask bbox;
- `focus_crop=false` keeps the full image;
- empty or disconnected masks disable focus crop automatically;
- `focus_crop_margin` expands the bbox in a normalized 1024-area model scale, so the visible model-space context stays similar across different source resolutions.

Use smaller margins for text/logo/detail repair, and larger margins when the model needs more surrounding context. Use `focus_crop_margin=0` when you want a tight mask bbox crop for a reference image.

## Reference Image Process

`RefineNode Reference Image Process` is a standalone image size alignment node:

- `fit_kontext_size=true` makes image1 choose the nearest Flux/Kontext-style preferred size;
- `fit_kontext_size=false` uses a `1024 * 1024` target area and rounds width/height to multiples of `8`;
- image1, optional image2, and optional image3 are all resized to the same target size;
- `resize_method` is shared by all connected images;
- `crop_mode=crop` uses center crop behavior;
- `crop_mode=disable` resizes directly without center cropping;
- `crop_mode=fill` preserves aspect ratio, scales to fit inside the target, then pads with black.

When `info` is connected, the output `info` stores the transform needed by `RefineNode Paste Back` to reverse this image-size processing before compositing.

## Example Workflows

`example_workflows/Reference-based Logo Refinement.json`

- uses a target image mask and a separate clean reference image;
- uses one `RefineNode Preprocess Mask` for the reference crop and another for the target mask;
- uses `RefineNode Reference Image Process` to align target, reference, and spatial mask;
- uses ComfyUI's native `TextEncodeQwenImageEditPlus` for Qwen conditioning.

`example_workflows/Reference-free Text Refinement.json`

- uses only a target image and painted text mask;
- sends target image plus spatial mask to the native Qwen text encoder;
- uses `RefineNode Paste Back` to composite the generated text repair into the original image.

The examples use Qwen Image Edit Plus because it is convenient for reference-guided detail repair, but the RefineNode preprocessing and paste-back nodes are not Qwen-specific.

## Model Encoding

RefineNode no longer includes a Qwen text encoder or model compatibility node. For Qwen Image Edit Plus workflows, use ComfyUI's built-in `TextEncodeQwenImageEditPlus`, shown in the UI as `文本编码（QwenImageEditPlus）`.

Model-specific nodes remain responsible for prompt encoding, image conditioning, reference latents, sampling, and model patches. RefineNode stays focused on model-independent image/mask preprocessing and paste-back.

## Notes

- No color transfer, color alignment, AdaIN, wavelet, or other color correction is applied.
- No model wrapper, text encoder wrapper, or attention/prompt patch is applied.
- If color shift or pixel offset appears, compare your model's generated size with `RefineNode Reference Image Process` output and make sure `RefineNode Paste Back` receives the matching processed `info`.
