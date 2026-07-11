# 阶段二需求文档：纯前向细粒度语义区域定位

## 1. 阶段目标

在单张测试图像到达时，利用阶段一生成的强文本锚点和 CLIP 视觉编码器内部空间 token，前向计算语义激活热力图与二值语义掩码，为阶段三的区域引导裁剪提供约束。

本阶段的核心要求是：**不使用梯度、不引入额外重型模型、不改变 CLIP 主干结构**。

## 2. 当前项目现状

当前 TPT 流程位于 `tpt_classification.py`：

- TPT 模式下，`data.datautils.AugMixAugmenter` 生成原图中心裁剪视图和若干随机增强视图；
- `test_time_tuning()` 对增强视图 logits 做低熵样本选择和边缘熵优化；
- `clip/custom_clip.py` 的 `ClipTestTimeTuning.inference()` 默认只返回全局图像特征经过文本分类器后的 logits。

现有代码没有暴露 ViT patch token，也没有语义热力图机制。因此阶段二需要在不破坏现有 RN/ViT 推理的前提下，新增可选的空间特征提取能力。

## 3. 输入与输出

### 3.1 输入

1. 原始 PIL 图像或未增强图像张量；
2. CLIP 模型及其视觉编码器；
3. 阶段一生成的强文本锚点：`anchors: Tensor[K, D]`；
4. 当前数据集类别顺序；
5. 定位参数：
   - 热力图 top 比例，例如 `mask_top_ratio=0.3`；
   - 最小掩码面积；
   - 平滑/插值策略；
   - 伪目标类别选择策略。

### 3.2 输出

1. 伪目标类别索引：`target_idx`；
2. 语义热力图：`heatmap: Tensor[H, W]`，范围建议归一化到 `[0, 1]`；
3. 二值语义掩码：`mask: Tensor[H, W]` 或 `PIL/np.ndarray`；
4. 可选调试信息：
   - 初始零样本概率 `p0`；
   - patch 级相似度 `S`；
   - mask 面积占比。

## 4. 功能需求

### R1. 初始伪目标类别选择

系统需要先对原图进行一次标准 CLIP 前向，得到全局图像特征 `v_cls`，再与所有强文本锚点计算相似度：

\[
p_0 = \mathrm{softmax}(s \cdot v_{cls} T_{anchor}^\top)
\]

其中 `s` 为 CLIP logit scale 或可配置温度。伪目标类别为：

\[
c_{target}=\arg\max_k p_{0,k}
\]

### R2. 空间 token 提取

对于 ViT 类 CLIP 视觉编码器，必须提取 patch/spatial tokens：

\[
V_{spatial}=[v_{p_1},...,v_{p_M}] \in \mathbb{R}^{M \times D}
\]

要求：

- 不修改 CLIP 预训练权重；
- 不对视觉编码器执行反向传播；
- 返回的空间 token 需投影到与文本锚点一致的维度 `D`；
- 若当前 `arch=RN50` 无自然 patch token，应明确处理策略：
  - 推荐优先支持 ViT 架构；
  - RN 架构可暂时回退到中心区域或提示用户切换 ViT。

### R3. patch-文本相似度计算

对伪目标文本锚点 `t_target` 和每个空间 token 计算余弦相似度：

\[
S(j)=\frac{v_{p_j}\cdot t_{target}}{\lVert v_{p_j}\rVert_2\lVert t_{target}\rVert_2}
\]

相似度向量需 reshape 为二维 patch 网格，例如 `14 x 14`。

### R4. 热力图上采样与归一化

- 将 patch 相似度网格通过双线性插值上采样到原图或裁剪前图像尺寸；
- 对 heatmap 做 min-max 归一化；
- 若最大值与最小值过近，需启用安全回退，避免除零。

### R5. 动态阈值二值化

根据 `mask_top_ratio` 仅保留响应最高的区域。例如 top 30%：

- 阈值为 heatmap 的 70 分位数；
- 大于等于阈值的位置置 1，其余置 0。

需保证：

- 掩码面积不得小于 `min_mask_area_ratio`；
- 若掩码过小或全空，回退为中心区域或全图掩码；
- 生成的 mask 与原图坐标系一致。

### R6. 可视化调试能力

建议支持保存：

- 原图；
- heatmap 伪彩色图；
- mask 覆盖图。

该能力仅用于调试，不应默认影响正式评估速度。

## 5. 非功能需求

1. **纯前向**：定位过程不得调用 `backward()` 或依赖梯度 hook。
2. **低开销**：除原有推理外，只允许轻量矩阵乘、reshape 和插值。
3. **可选启用**：通过参数启用 SAR-TPT 定位，不影响原始 TPT 复现实验。
4. **架构可控**：优先支持 `ViT-B/16`、`ViT-L/14`；RN 支持可作为后续扩展。
5. **类别顺序安全**：锚点矩阵行顺序必须与当前 `model.reset_classnames()` 后的类别顺序一致。

## 6. 建议涉及文件

未来实现时建议新增：

- `utils/semantic_region.py`
- `utils/visualization.py`（可选）

可能需要轻量扩展：

- `clip/custom_clip.py`：增加可选 `encode_image_with_spatial_tokens()` 或类似接口；
- `tpt_classification.py`：在构造 TPT transform 前或 dataset transform 中传入定位器。

## 7. 验收标准

1. 对一张图像能输出 `target_idx`、`heatmap`、`mask`。
2. heatmap 尺寸与原图或指定输入尺寸一致。
3. mask 非空，面积比例在合理范围内。
4. 全流程不产生 prompt 或视觉编码器梯度。
5. 在 ViT 架构上，空间 token 与文本锚点维度一致。

## 8. 第二阶段实现与验收记录

本阶段已补充纯前向语义区域定位能力，交付内容如下：

- `clip/custom_clip.py`
  - 新增 `ClipTestTimeTuning.encode_image_features()`；
  - 新增 `ClipTestTimeTuning.encode_image_with_spatial_tokens()`；
  - 对 ViT CLIP 复刻视觉前向流程，返回：
    - `cls_feature: [B, D]`；
    - `spatial_tokens: [B, M, D]`；
    - `grid_size: (Gh, Gw)`；
  - 对 RN 类 backbone 显式抛出 `NotImplementedError`，符合阶段二“优先 ViT”的边界。

- `utils/semantic_region.py`
  - 新增 `SemanticRegionLocator`；
  - 新增 `SemanticRegionResult`；
  - 支持载入阶段一 anchor tensor 或 anchor payload；
  - 完成伪目标类别选择、patch-anchor 相似度计算、热力图上采样、动态阈值二值 mask、fallback 逻辑；
  - 全流程使用 `torch.no_grad()`，不产生梯度；
  - 提供 `save_semantic_region_debug()` 保存 image/heatmap/mask/meta 的 `.pt` 调试文件。

- `scripts/validate_semantic_region.py`
  - 阶段二验收脚本；
  - 输入单张图片和阶段一 `.pt` 锚点；
  - 输出定位结果统计并保存 debug tensor。

- `tests/test_stage2_semantic_region.py`
  - 不加载 CLIP、不依赖真实数据；
  - 使用 fake model 验证 locator 合同；
  - 覆盖 heatmap normalize、mask threshold、patch upsample 等基础逻辑。

### 8.1 阶段二核心接口

在后续阶段三中，可以这样调用：

```python
from utils.semantic_region import SemanticRegionLocator
from utils.text_anchors import load_text_anchor_file

anchor_payload = load_text_anchor_file("assets/anchors/features/Pets_ViT-B-16.pt")
locator = SemanticRegionLocator(anchor_payload, mask_top_ratio=0.3)
result = locator.locate(model, image_tensor)

heatmap = result.heatmap  # [H, W]
mask = result.mask        # [H, W], bool
target_idx = result.target_idx
```

其中 `model` 必须提供：

```python
model.encode_image_with_spatial_tokens(image_tensor)
```

当前 `ClipTestTimeTuning` 已提供该接口。

### 8.2 阶段二验收命令

在具备依赖、CLIP 权重、阶段一 anchors 和样例图片的环境中执行：

```bash
python scripts/validate_semantic_region.py \
  --image /path/to/sample.jpg \
  --anchor-path assets/anchors/features/Pets_ViT-B-16.pt \
  --dataset Pets \
  --arch ViT-B/16 \
  --gpu 0 \
  --debug-prefix outputs/stage2_debug/pets_sample
```

预期输出：

- `target_idx`；
- `mask_area_ratio`；
- `heatmap_shape`；
- `mask_shape`；
- `patch_similarity_shape`；
- 是否触发 fallback。

同时会保存：

- `outputs/stage2_debug/pets_sample_image.pt`
- `outputs/stage2_debug/pets_sample_heatmap.pt`
- `outputs/stage2_debug/pets_sample_mask.pt`
- `outputs/stage2_debug/pets_sample_meta.pt`

### 8.3 纯工具测试

在可运行环境中执行：

```bash
python -m unittest tests/test_stage2_semantic_region.py
```

该测试不下载 CLIP、不读取数据集，可用于快速验证阶段二工具逻辑。

### 8.4 验收标准对应关系

- 输出 `target_idx`、`heatmap`、`mask`：由 `SemanticRegionLocator.locate()` 返回；
- heatmap 尺寸与输入图像一致：`patch_similarity_to_heatmap(..., output_size=image.shape[-2:])`；
- mask 非空及面积兜底：`mask_from_heatmap()` 支持 `center/full/none` fallback；
- 不产生梯度：`locate()` 内部使用 `torch.no_grad()`；
- ViT 空间 token 与文本锚点同维度：`encode_image_with_spatial_tokens()` 对 patch token 应用 `ln_post` 和 `proj`；
- RN fallback：默认显式报错，可通过 `allow_non_vit_fallback=True` 获得中心 fallback mask。
