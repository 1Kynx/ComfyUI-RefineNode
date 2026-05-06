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
