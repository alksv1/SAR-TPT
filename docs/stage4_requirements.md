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
