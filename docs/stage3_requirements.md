# 阶段三需求文档：区域引导的约束多视图数据增强

## 1. 阶段目标

基于阶段二生成的语义掩码，替代或增强当前项目中的随机全局增强流程，使 TPT 使用的多视图样本在保持随机性和多样性的同时，必须覆盖关键细粒度语义区域，降低随机裁剪破坏判别区域的概率。

本阶段是 SAR-TPT 相比原始 TPT 的输入端核心改造。

## 2. 当前项目现状

当前多视图增强在 `data/datautils.py` 中实现：

- `AugMixAugmenter.__call__(x)` 返回 `[image] + views`；
- 原图视图来自 `base_transform`，通常为 Resize + CenterCrop；
- 增强视图由 `augmix()` 生成；
- `augmix()` 内部 `get_preaugment()` 使用：
  - `RandomResizedCrop(224)`；
  - `RandomHorizontalFlip()`。

问题是 `RandomResizedCrop` 不关心局部语义区域，可能裁掉细粒度关键部件。

## 3. 输入与输出

### 3.1 输入

1. 原始 PIL 图像 `x`；
2. 阶段二输出的语义掩码 `M_sem`；
3. 原有 preprocess，包括 ToTensor 与 CLIP normalize；
4. 增强参数：
   - `n_views`；
   - `crop_scale`；
   - `crop_ratio`；
   - 覆盖率阈值 `tau_cov`；
   - 最大重采样次数 `max_crop_trials`；
   - 是否启用 AugMix；
   - 随机水平翻转概率。

### 3.2 输出

与现有 TPT 保持一致：

```python
[clean_image, guided_view_1, guided_view_2, ..., guided_view_N]
```

其中所有元素均为已经完成 CLIP normalize 的 tensor，便于 `tpt_classification.py` 继续执行：

```python
images = torch.cat(images, dim=0)
```

## 4. 功能需求

### R1. 区域引导候选裁剪

每次生成增强视图时，先随机采样候选裁剪框 `B_cand`，然后计算其对语义掩码的覆盖率：

\[
Coverage(B_{cand}, M_{sem}) = \frac{\sum_{(x,y)\in B_{cand}}M_{sem}(x,y)}{\sum_{all}M_{sem}(x,y)}
\]

当 `Coverage >= tau_cov` 时接受该裁剪框，否则重新采样。

### R2. 最大尝试次数与回退机制

为避免极端图像导致死循环，必须设置 `max_crop_trials`。

达到上限仍无合格裁剪框时，按以下优先级回退：

1. 使用语义掩码外接框并加入随机 padding；
2. 使用中心裁剪；
3. 使用原始 `RandomResizedCrop`。

回退事件应可统计，用于后续实验分析。

### R3. 增强视图多样性

区域引导不应退化为固定裁剪。即使所有裁剪都覆盖语义区域，也应保留：

- 随机尺度；
- 随机长宽比；
- 随机水平翻转；
- 可选 AugMix 颜色/纹理扰动。

### R4. 与原有 AugMixAugmenter 接口兼容

为降低改造成本，新增强器应尽量保持当前接口：

```python
augmenter = SomeAugmenter(base_transform, preprocess, n_views, ...)
views = augmenter(pil_image)
```

若需要阶段二定位结果，可以采用以下设计之一：

1. 在 augmenter 内部持有 `semantic_locator`，每次 `__call__` 先定位再增强；
2. dataset 返回原图路径或 PIL 图像，由主循环先定位后生成 views；
3. 新建 SAR 专用 dataset transform，封装定位和增强。

推荐优先选择方案 1，改动面最小。

### R5. 原图视图保持稳定

返回列表第一个元素仍应是稳定的 clean/center crop 视图，用于最终推理或与当前 TPT 行为保持兼容。区域引导主要作用于后续增强视图。

### R6. 坐标一致性

如果阶段二的 mask 是基于 resize 后图像或 center crop 后图像生成的，阶段三必须明确裁剪坐标系：

- 推荐在同一 PIL 原图坐标系下生成 mask 和 crop box；
- 若使用 CLIP 输入分辨率坐标系，需要记录从原图到输入图的缩放关系；
- 覆盖率计算不得混用不同尺寸坐标。

## 5. 参数建议

初始实验可采用：

| 参数 | 建议值 | 说明 |
|---|---:|---|
| `n_views` | `batch_size - 1` | 与原始 TPT 对齐 |
| `tau_cov` | `0.6` | 保证关键区域覆盖 |
| `max_crop_trials` | `10` 或 `20` | 控制增强耗时 |
| `mask_top_ratio` | `0.3` | 与阶段二联动 |
| `crop_scale` | `(0.5, 1.0)` | 避免裁剪过小 |
| `crop_ratio` | `(3/4, 4/3)` | 沿用 torchvision 默认思路 |

## 6. 非功能需求

1. **兼容原始 TPT**：未启用 SAR-TPT 时，原有 `AugMixAugmenter` 行为不变。
2. **速度可控**：重采样次数和定位次数要受控，不能让单样本延迟大幅超过 TPT。
3. **确定性可选**：设置随机种子后，增强采样应尽量可复现。
4. **失败可观测**：需要统计平均重采样次数、回退次数、平均 coverage。

## 7. 建议涉及文件

未来实现时建议新增：

- `data/sar_augment.py`

可能修改：

- `data/datautils.py`：注册 SAR 增强器或复用部分 AugMix 逻辑；
- `tpt_classification.py`：根据参数选择 `AugMixAugmenter` 或 SAR 增强器。

## 8. 验收标准

1. SAR 增强器输出格式与原始 TPT 一致。
2. 每个增强视图的裁剪框 coverage 默认不低于 `tau_cov`，除非触发回退。
3. 回退不会导致程序中断。
4. 禁用 SAR 参数时，原始 TPT 实验可正常运行。
5. 可记录并打印 coverage、重采样次数等调试统计。
