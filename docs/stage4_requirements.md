# 阶段四需求文档：双模态一致性过滤与边缘熵优化

## 1. 阶段目标

在阶段三生成高质量区域引导增强视图后，对每个视图进行可靠性评估，只保留同时满足预测置信度和文本锚点语义对齐要求的视图，再使用可靠视图集合执行边缘熵最小化，更新测试时可学习提示。

本阶段是 SAR-TPT 的优化端防崩溃机制，目标是减少伪标签漂移、过度自信错误预测和类别崩溃。

## 2. 当前项目现状

当前 `tpt_classification.py` 中的 TPT 优化逻辑包括：

1. `test_time_tuning()` 对输入多视图执行模型前向；
2. 首轮调用 `select_confident_samples(output, args.selection_p)`；
3. 该函数仅按预测熵从低到高选取前 `selection_p` 比例；
4. `avg_entropy(output)` 计算边缘熵；
5. 对 prompt 参数执行 1 至多步 AdamW 更新。

现有问题：

- 只依赖 logits 熵，无法识别过度自信但语义偏离的错误视图；
- 没有利用阶段一强文本锚点；
- selected_idx 在多步 TTA 中固定，但筛选标准单一。

## 3. 输入与输出

### 3.1 输入

1. 阶段三生成的多视图 tensor：`inputs: Tensor[N, C, H, W]`；
2. 当前模型输出 logits：`logits: Tensor[N, K]`；
3. 每个增强视图的视觉特征：`v_i: Tensor[N, D]`；
4. 阶段一强文本锚点：`anchors: Tensor[K, D]`；
5. 阶段二确定的伪目标类别 `target_idx`；
6. 过滤参数：
   - `lambda_anchor`；
   - `entropy_scale`；
   - `reliable_top_k` 或 `reliable_ratio`；
   - 最小保留视图数 `min_reliable_views`。

### 3.2 输出

1. 可靠视图索引：`reliable_idx`；
2. 可靠视图 logits：`reliable_logits`；
3. SAR 边缘熵损失：`loss_sar`；
4. 可选统计：
   - 平均预测熵；
   - 平均锚点相似度；
   - 过滤前后视图数量；
   - 被过滤视图比例。

## 4. 功能需求

### R1. 预测熵计算

对每个视图的概率分布计算香农熵：

\[
H(p_i)=-\sum_{k=1}^{K}p_{i,k}\log p_{i,k}
\]

其中：

```python
p_i = softmax(logits_i)
```

熵越低表示模型越自信。

### R2. 锚点相似度计算

对每个增强视图的视觉特征和伪目标类别强文本锚点计算余弦相似度：

\[
CosSim(v_i,t^{anchor}_{target})=\frac{v_i\cdot t^{anchor}_{target}}{\lVert v_i\rVert_2\lVert t^{anchor}_{target}\rVert_2}
\]

要求：

- 视觉特征必须与锚点处于同一 CLIP embedding 空间；
- 特征与锚点均需 L2 normalize；
- 不应对锚点求梯度。

### R3. 双模态综合评分

每个视图综合评分定义为：

\[
Score_i = \lambda \cdot CosSim(v_i,t^{anchor}_{target}) - (1-\lambda) \cdot \alpha \cdot H(p_i)
\]

其中：

- `lambda` 对应参数 `lambda_anchor`；
- `alpha` 对应 `entropy_scale`，用于统一量纲；
- 分数越高，视图越可靠。

### R4. 可靠视图选择

支持两种选择策略：

1. 比例选择：保留 top `reliable_ratio`；
2. 数量选择：保留 top `reliable_top_k`。

必须保证：

- 保留数量不少于 `min_reliable_views`；
- 保留数量不超过视图总数；
- 若所有评分异常，回退到原始 TPT 的低熵选择策略。

### R5. SAR 边缘熵损失

仅使用可靠视图集合计算平均概率分布：

\[
\bar{p}=\frac{1}{|V_{reliable}|}\sum_{i\in V_{reliable}}p_i
\]

SAR 损失为：

\[
\mathcal{L}_{SAR}=-\sum_k\bar{p}_k\log\bar{p}_k
\]

该损失用于更新 prompt 参数，不更新 CLIP 图像编码器、文本编码器和强文本锚点。

### R6. Episodic TTA 重置

必须保留当前项目的 episodic 设计：

- 每张测试图像适应前，prompt reset 到初始状态；
- optimizer state reset 到初始状态；
- 当前样本优化结束后，不把 prompt 状态传递到下一张图像。

这与 `test_time_adapt_eval()` 中现有逻辑一致，未来实现不得破坏。

### R7. 与 CoOp/CoCoOp 的关系

第一版 SAR-TPT 推荐优先支持非 CoCoOp 分支，即 `get_coop()` + 可学习 prompt。

CoCoOp 分支中 prompt generator 和 `pgen_ctx` 的更新路径不同，可作为后续扩展。文档要求：

- SAR 过滤逻辑应设计为可复用函数；
- 若暂不支持 CoCoOp，命令行应明确报错或自动回退原始 TPT。

## 5. 非功能需求

1. **数值稳定**：计算 `log` 时需加入 epsilon 或使用 log-softmax 形式。
2. **低额外显存**：锚点矩阵只读缓存，不随视图重复复制大张量。
3. **可配置**：过滤强度可通过命令行参数调节。
4. **可消融**：必须支持关闭锚点相似度，仅用熵；关闭熵，仅用锚点相似度；关闭过滤。
5. **可复现实验**：输出过滤统计，便于论文消融分析。

## 6. 建议命令行参数

未来实现时建议新增：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--sar_tpt` | `False` | 启用 SAR-TPT |
| `--anchor_path` | `None` | 阶段一锚点文件路径 |
| `--lambda_anchor` | `0.5` | 双模态评分权重 |
| `--entropy_scale` | `1.0` | 熵项缩放 |
| `--reliable_ratio` | `0.5` | 保留视图比例 |
| `--min_reliable_views` | `1` | 最小保留数 |
| `--disable_anchor_filter` | `False` | 消融：禁用锚点过滤 |
| `--disable_entropy_filter` | `False` | 消融：禁用熵过滤 |

## 7. 建议涉及文件

未来实现时建议新增：

- `utils/sar_filter.py`

可能修改：

- `clip/custom_clip.py`：提供同时返回 logits 和 image features 的接口；
- `tpt_classification.py`：替换或扩展 `select_confident_samples()` 与 `test_time_tuning()`。

## 8. 验收标准

1. 对一批增强视图能输出可靠索引和 SAR loss。
2. SAR loss 可正常反向传播到 prompt 参数。
3. CLIP 图像编码器、文本编码器和锚点不产生梯度更新。
4. 关闭 SAR 参数时，原始 TPT 行为保持不变。
5. 支持至少三类消融：无引导裁剪、单熵过滤、单锚点过滤。

## 9. 第四阶段实现与验收记录

本阶段已完成双模态一致性过滤与 SAR 边缘熵优化，并接入主 TPT 测试时优化循环。至此，SAR-TPT 从阶段一 anchors、阶段二语义定位、阶段三区域引导增强到阶段四可靠视图优化已形成完整闭环。

### 9.1 交付文件

- `utils/sar_filter.py`
  - `prediction_entropy()`：逐视图预测熵；
  - `compute_anchor_target()`：基于 clean view 与强文本锚点选择伪目标类别；
  - `dual_modality_scores()`：计算锚点相似度与预测熵的综合可靠性分数；
  - `select_reliable_views()`：按 score 选择可靠视图；
  - `marginal_entropy_loss()`：可靠视图集合上的边缘熵损失；
  - `sar_filter_and_loss()`：阶段四一站式入口；
  - `SARFilterResult`：保存可靠索引、score、entropy、anchor similarity 和调试信息。

- `clip/custom_clip.py`
  - 新增 `inference_with_features()`；
  - 冻结视觉编码器提取 normalized image features；
  - logits 仍由当前可学习 prompt 的 text features 产生，因此 SAR loss 可反向传播到 prompt 参数。

- `tpt_classification.py`
  - `test_time_tuning()` 已接入 SAR filter；
  - `--sar_tpt` 且存在当前 anchor payload 时，使用双模态过滤与 SAR loss；
  - 支持消融参数关闭过滤或单独关闭锚点/熵分量。

- `tests/test_stage4_sar_filter.py`
  - 不加载 CLIP、不依赖数据集；
  - 验证 entropy、target selection、score、reliable view selection、loss backward。

- `scripts/validate_sar_filter.py`
  - 使用 synthetic tensors 验收阶段四过滤与 loss backward。

### 9.2 主流程参数

新增阶段四参数：

```bash
--lambda_anchor 0.5
--entropy_scale 1.0
--reliable_ratio 0.5
--reliable_top_k 0
--min_reliable_views 1
--disable_sar_filter
--disable_anchor_filter
--disable_entropy_filter
```

说明：

- `--disable_sar_filter`：完全关闭第四阶段过滤，退回原始 TPT 熵过滤，但仍可保留阶段三 guided crop；
- `--disable_anchor_filter`：只用预测熵过滤；
- `--disable_entropy_filter`：只用锚点相似度过滤；
- `--reliable_top_k` 大于 0 时优先于 `--reliable_ratio`。

### 9.3 完整 SAR-TPT 示例

```bash
python tpt_classification.py /path/to/data \
  --test_sets Pets \
  -a ViT-B/16 \
  -b 64 \
  --gpu 0 \
  --tpt \
  --sar_tpt \
  --anchor_path assets/anchors/features/Pets_ViT-B-16.pt \
  --tau_cov 0.6 \
  --max_crop_trials 20 \
  --lambda_anchor 0.5 \
  --entropy_scale 1.0 \
  --reliable_ratio 0.5
```

### 9.4 消融实验命令示例

仅关闭阶段四，保留阶段三 guided crop：

```bash
--sar_tpt --disable_sar_filter
```

仅用预测熵过滤：

```bash
--sar_tpt --disable_anchor_filter
```

仅用锚点相似度过滤：

```bash
--sar_tpt --disable_entropy_filter
```

关闭 guided crop、只跑原始 TPT：

```bash
--tpt
```

### 9.5 阶段四独立验收

不依赖 CLIP 的 synthetic 验收：

```bash
python scripts/validate_sar_filter.py \
  --num-views 8 \
  --num-classes 4 \
  --dim 16 \
  --reliable-ratio 0.5 \
  --lambda-anchor 0.5
```

纯工具测试：

```bash
python -m unittest tests/test_stage4_sar_filter.py
```

### 9.6 验收标准对应关系

- 预测熵计算：`prediction_entropy()`；
- 锚点相似度计算：`dual_modality_scores()`；
- 双模态综合评分：`Score = lambda * sim - (1-lambda) * alpha * entropy`；
- 可靠视图选择：`select_reliable_views()` 支持 ratio/top-k/min views；
- SAR 边缘熵损失：`marginal_entropy_loss()`；
- Episodic TTA：保留 `test_time_adapt_eval()` 中每样本 `model.reset()` 和 optimizer state reset；
- CoCoOp 边界：`--sar_tpt --cocoop` 显式报错；
- 消融支持：`--disable_sar_filter`、`--disable_anchor_filter`、`--disable_entropy_filter`。
