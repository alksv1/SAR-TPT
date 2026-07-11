# SAR-TPT 分阶段实现路线与实验需求汇总

## 1. 总体目标

在当前 TPT 代码库上实现 `design.md` 提出的 SAR-TPT：

1. 阶段一：离线 LLM 强文本锚点构建；
2. 阶段二：纯前向细粒度语义区域定位；
3. 阶段三：区域引导约束多视图增强；
4. 阶段四：双模态一致性过滤与边缘熵优化。

实现原则：

- 不修改 CLIP 主干权重；
- 默认保持原始 TPT 可复现；
- SAR-TPT 通过显式参数启用；
- 优先支持 ViT 架构和非 CoCoOp 分支；
- 每个阶段均需可单独消融。

## 2. 当前代码基线

主要文件职责：

| 文件 | 当前职责 | SAR-TPT 相关性 |
|---|---|---|
| `tpt_classification.py` | 主评估入口、TPT 优化循环 | 阶段三/四集成入口 |
| `clip/custom_clip.py` | CLIP + PromptLearner 封装 | 需暴露文本/视觉特征与空间 token |
| `data/datautils.py` | 数据集构建、AugMix 多视图增强 | 阶段三替换或扩展增强器 |
| `data/cls_to_names.py` | 细粒度数据集类别名 | 阶段一类别来源 |
| `data/imagenet_variants.py` | ImageNet 子集标签映射 | 阶段一/四类别顺序安全 |
| `utils/tools.py` | 指标、日志、模型加载 | 可复用统计工具 |

## 3. 推荐实现顺序

### Step 1：实现阶段一资产生成

先完成强文本锚点离线文件，原因：

- 阶段二和阶段四都依赖锚点；
- 可独立验证类别顺序和特征维度；
- 不影响当前 TPT 主流程。

最低验收：对 `Pets`、`Cars`、`Aircraft` 至少一个数据集生成 `anchors.pt`。

### Step 2：实现阶段二定位并做可视化验证

在 ViT 架构下验证热力图是否聚焦目标部位。该阶段建议先用少量图片保存 overlay，不急于跑完整准确率。

最低验收：输入单张 PIL 图像输出非空 mask。

### Step 3：实现阶段三 SAR 增强器

将语义 mask 注入裁剪逻辑，保持输出格式与当前 `AugMixAugmenter` 一致。

最低验收：`tpt_classification.py` 中 `images = torch.cat(images, dim=0)` 不需要大改即可运行。

### Step 4：实现阶段四过滤和 SAR loss

将原始 `select_confident_samples()` 扩展为双模态过滤，保留原始熵过滤作为回退和消融。

最低验收：loss 可反向传播至 prompt，且原始 TPT 可通过参数恢复。

## 4. 实验需求

### 4.1 主实验

推荐数据集：

- 细粒度：`Pets`、`Cars`、`Aircraft`；
- 分布偏移：`A`、`R`、`V`；
- 若后续加入 `CUB-200-2011`，需先补充 dataset builder 和类别名。

指标：

- Top-1 Accuracy；
- Top-5 Accuracy；
- 单样本平均推理时间；
- 可选 ECE。

### 4.2 消融实验

必须支持：

1. 原始 TPT；
2. SAR-TPT 完整版；
3. 无区域引导裁剪，仅双模态过滤；
4. 有区域引导裁剪，仅熵过滤；
5. 有区域引导裁剪，仅锚点相似度过滤；
6. LLM 强锚点替换为普通模板锚点。

### 4.3 效率实验

需记录：

- 每张图平均定位耗时；
- 平均 crop 重采样次数；
- 平均 TTA 优化耗时；
- 总推理延迟。

## 5. 风险与约束

1. **RN50 无 patch token**：SAR 定位优先支持 ViT，RN50 可作为原始 TPT 对照。
2. **类别顺序错位**：所有 anchors 必须随 `reset_classnames()` 的类别顺序重排。
3. **LLM 描述质量不稳定**：描述文件需允许人工修订和版本化。
4. **增强器内定位成本**：如果每个 view 重复定位会过慢，应每张原图只定位一次。
5. **CoCoOp 分支复杂**：第一版可不支持，但需要清晰回退。

## 6. 文档索引

- 阶段一：`docs/stage1_requirements.md`
- 阶段二：`docs/stage2_requirements.md`
- 阶段三：`docs/stage3_requirements.md`
- 阶段四：`docs/stage4_requirements.md`

## 7. 第一阶段交付状态

第一阶段已开工并完成文档级验收入口：

- 新增 `utils/text_anchors.py`：类别顺序解析、描述 JSON 校验、锚点 `.pt` 校验。
- 新增 `scripts/build_text_anchors.py`：离线描述缓存准备与 CLIP 文本锚点编码。
- 新增 `scripts/validate_text_anchors.py`：阶段一验收脚本。
- 新增 `tests/test_stage1_text_anchors.py`：不加载 CLIP 的纯 Python 单元测试。
- 新增 `assets/anchors/descriptions/` 与 `assets/anchors/features/` 作为默认资产目录。

注意：本地未运行代码。后续在可运行环境中先按 `docs/stage1_requirements.md` 的“第一阶段实现与验收记录”执行验收命令，再进入阶段二。

### 7.1 第一阶段补充：OpenAI-compatible LLM 描述生成

`build_text_anchors.py` 已支持通过 `--llm-generate` 调用 OpenAI-compatible Chat Completions 接口生成类别描述。默认仍是离线 fallback；只有显式传入 `--llm-generate` 才会联网请求。

关键参数：

- `--llm-base-url`：兼容服务 base URL，例如 `https://api.openai.com/v1`。
- `--llm-model`：描述生成模型名。
- `--llm-api-key-env`：API key 环境变量名，默认 `OPENAI_API_KEY`。
- `--llm-temperature`：默认 `0.2`。
- `--llm-retries`：每个类别失败重试次数。
- `--llm-save-every`：每生成多少个类别保存一次 JSON，默认每类保存，便于断点续跑。

该功能只负责生成描述 JSON；后续仍通过 CLIP 文本编码器生成强文本锚点。

## 8. 第二阶段交付状态

第二阶段已完成编码入口：

- `clip/custom_clip.py` 已支持 ViT 空间 token 提取；
- `utils/semantic_region.py` 已实现纯前向语义区域定位；
- `scripts/validate_semantic_region.py` 已提供单图验收入口；
- `tests/test_stage2_semantic_region.py` 已提供无需 CLIP 权重的工具测试。

下一阶段应在 `data/sar_augment.py` 中接入 `SemanticRegionLocator` 的 `mask` 输出，实现区域引导裁剪。

## 9. 第三阶段交付状态

第三阶段已完成区域引导多视图增强：

- 新增 `data/sar_augment.py`，实现 semantic mask coverage 约束裁剪；
- `tpt_classification.py` 已支持 `--sar_tpt` 和阶段三裁剪参数；
- 新增 `scripts/validate_sar_augment.py` 做不依赖 CLIP 的增强器验收；
- 新增 `tests/test_stage3_sar_augment.py` 做纯工具测试。

下一阶段应实现双模态一致性过滤与 SAR loss，将阶段一 anchors 和阶段二 target anchor 用于视图可靠性筛选。

## 10. 第四阶段交付状态与项目闭环

第四阶段已完成：

- 新增 `utils/sar_filter.py`，实现双模态一致性过滤与 SAR loss；
- `clip/custom_clip.py` 新增 `inference_with_features()`，用于同时返回 logits 和冻结 image features；
- `tpt_classification.py` 的 `test_time_tuning()` 已接入 SAR loss；
- 新增 `scripts/validate_sar_filter.py` 和 `tests/test_stage4_sar_filter.py`。

当前项目已具备完整 SAR-TPT 流水线：

1. `scripts/build_text_anchors.py` 生成/编码强文本锚点；
2. `utils/semantic_region.py` 基于 ViT spatial tokens 生成 semantic mask；
3. `data/sar_augment.py` 根据 mask 生成区域引导多视图；
4. `utils/sar_filter.py` 对多视图执行双模态过滤并计算 SAR loss；
5. `tpt_classification.py --sar_tpt` 串联完整流程。

推荐验收顺序：

```bash
python -m unittest tests/test_stage1_text_anchors.py
python -m unittest tests/test_stage2_semantic_region.py
python -m unittest tests/test_stage3_sar_augment.py
python -m unittest tests/test_stage4_sar_filter.py
```

完整实验前需要先生成对应数据集和架构的 anchors：

```bash
python scripts/build_text_anchors.py \
  --dataset Pets \
  --arch ViT-B/16 \
  --llm-generate \
  --description-path assets/anchors/descriptions/Pets.json \
  --output assets/anchors/features/Pets_ViT-B-16.pt
```

然后运行：

```bash
python tpt_classification.py /path/to/data \
  --test_sets Pets \
  -a ViT-B/16 \
  -b 64 \
  --gpu 0 \
  --tpt \
  --sar_tpt \
  --anchor_path assets/anchors/features/Pets_ViT-B-16.pt
```
