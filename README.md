# ComfyUI-RefineNode

ComfyUI-RefineNode 是一组用于局部重绘、细节修复、产品/logo/文字细化和结果贴回的通用 ComfyUI 节点。

插件参考了 [RefineAnything](https://github.com/limuloo/RefineAnything) 项目的图像局部细化流程，并将其中适合 ComfyUI 工作流复用的图像、遮罩、参考图预处理和 paste-back 部分整理为独立节点。

相关项目：

- GitHub: [limuloo/RefineAnything](https://github.com/limuloo/RefineAnything)
- Hugging Face: [limuloo1999/RefineAnything](https://huggingface.co/limuloo1999/RefineAnything)

## Installation

进入 ComfyUI 的 `custom_nodes` 目录后克隆本仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1Kynx/ComfyUI-RefineNode.git
```

然后重启 ComfyUI。

## Nodes

| Node | Output | Purpose |
|------|--------|---------|
| `RefineNode Preprocess Mask` | `image`, `spatial_mask_image`, `mask`, `info` | 处理输入图像和可选遮罩，输出模型使用的图像、空间遮罩图、遮罩和贴回所需的 `REFINENODE_INFO`。 |
| `RefineNode Reference Image Process` | `image1`, `image2`, `image3`, `info` | 将一到三张图像处理到一致尺寸，并同步更新 `REFINENODE_INFO`，便于后续准确贴回。 |
| `RefineNode Paste Back` | `image`, `paste_mask` | 将生成结果按遮罩或裁剪区域贴回原图。 |

## Example Workflows

`example_workflows/Reference-based Logo Refinement.json`

- 参考图模式案例，用于根据参考图修复目标图中的 logo、文字或产品细节。
- 使用参考图模式时，确保细化图连接 `image1`，参考图连接 `image2`，细化图遮罩连 `image3`。

`example_workflows/Reference-free Text Refinement.json`

- 无参考图文字修复案例，用于根据目标图和遮罩直接细化局部文字。

## Node Details

### RefineNode Preprocess Mask

用于在进入模型前整理细化图和遮罩，并生成后续贴回需要的 `info`。

输入：

- `image`: 需要修复或作为参考的图像。
- `mask`: 可选遮罩；未连接时会使用空遮罩，节点不会报错。
- `focus_crop`: 开启后会围绕遮罩区域裁剪图像，减少模型需要处理的无关区域。
- `focus_crop_margin`: 裁剪时保留的额外上下文范围。
- `spatial_prompt_source`: 控制 `spatial_mask_image` 的来源，`mask` 使用原遮罩形状，`bbox` 使用遮罩外接矩形。

输出：

- `image`: 处理后的模型输入图。
- `spatial_mask_image`: 给 Qwen Image Edit Plus 等图像编辑节点使用的空间遮罩图。
- `mask`: 与输出 `image` 对齐的遮罩。
- `info`: 保存原图、裁剪位置、遮罩和贴回坐标的数据，必须传给后续需要贴回的节点。

### RefineNode Reference Image Process

用于把细化图、参考图和遮罩图处理到完全一致的尺寸，适合多图参考或 Qwen Image Edit Plus 这类需要多张图像输入的工作流。

输入：

- `image1`: 必填，通常连接细化图。
- `image2`: 可选，通常连接参考图。
- `image3`: 可选，参考图模式中通常连接细化图遮罩，也就是 `spatial_mask_image`。
- `info`: 可选，来自 `RefineNode Preprocess Mask`；连接后会同步记录本节点的缩放、裁剪或填充信息，保证 `Paste Back` 能按正确位置贴回。
- `fit_kontext_size`: 开启时使用 ComfyUI/Flux Kontext 风格的目标尺寸；关闭时按约 `1024 x 1024` 目标面积并按 8 的倍数取整。
- `resize_method`: 三张图共用的缩放算法，默认 `lanczos`。
- `crop_mode`: `crop` 居中裁剪到目标比例，`disable` 直接缩放，`fill` 等比缩放后用黑色填充到目标尺寸。

输出：

- `image1`: 处理后的细化图。
- `image2`: 处理后的参考图；未连接输入时输出空白占位图。
- `image3`: 处理后的遮罩图或第三张参考图；未连接输入时输出空白占位图。
- `info`: 更新后的贴回信息。

参考图模式注意：

- 细化图连接 `image1`。
- 参考图连接 `image2`。
- 细化图遮罩连接 `image3`。

### RefineNode Paste Back

用于把模型生成结果贴回原图，输出最终修复图和实际使用的贴回遮罩。

输入：

- `generated_image`: 模型生成后的图像。
- `info`: 来自 `RefineNode Preprocess Mask` 或 `RefineNode Reference Image Process` 的贴回信息。
- `paste_back_mode`: `mask` 按原遮罩贴回，`bbox` 按遮罩外接矩形区域贴回。
- `mask_grow`: 贴回前扩张遮罩边缘，适合减少细碎边缘漏贴。
- `blend_blur`: 对贴回遮罩做模糊羽化，适合减轻边缘接缝。

输出：

- `image`: 贴回后的最终图像。
- `paste_mask`: 本次实际用于合成的遮罩，方便预览和排查贴回范围。
