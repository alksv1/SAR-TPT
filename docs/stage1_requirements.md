# 阶段一需求文档：离线 LLM 强文本锚点构建

## 1. 阶段目标

根据 `design.md` 中 SAR-TPT 的第一阶段设计，构建一个**离线生成、可复用、零测试时延迟**的强文本锚点系统。该系统针对每个目标数据集的类别集合，生成细粒度视觉属性描述，并通过冻结的 CLIP 文本编码器编码为类别级文本锚点，用于后续阶段的语义区域定位与双模态一致性过滤。

本阶段只负责数据与特征资产的准备，不参与测试时提示更新，也不改变现有 CLIP/TPT 主推理逻辑。

## 2. 背景与问题

当前项目来自 TPT 官方实现，默认提示主要依赖：

- 手工模板，例如 `a photo of a [CLASS]`；
- CoOp/CoCoOp 预训练软提示；
- `clip/custom_clip.py` 中 `PromptLearner` 动态拼接类别名。

这些提示在细粒度场景中缺乏局部属性描述，例如鸟类喙部、翼纹、汽车年款差异、飞机尾翼结构等。SAR-TPT 需要额外构建强文本锚点，使模型在后续局部区域定位和视图过滤时拥有更稳定的语义参照。

## 3. 输入与输出

### 3.1 输入

1. 数据集标识：与当前项目保持一致，例如：
   - `Pets`
   - `Cars`
   - `Aircraft`
   - 后续可扩展 `CUB`、`ImageNet-A`、`ImageNet-R`、`ImageNet-V2` 等。
2. 类别名称列表：
   - 现有来源包括 `data/cls_to_names.py`、`data/imagnet_prompts.py`、`data/fewshot_datasets.py`。
3. LLM 生成配置：
   - 每类生成描述数量；
   - 描述语言；
   - 是否启用人工缓存文件；
   - 是否跳过已存在类别。
4. CLIP 文本编码器配置：
   - `arch`，需与实验推理阶段一致，例如 `RN50`、`ViT-B/16`、`ViT-L/14`；
   - 文本模板策略。

### 3.2 输出

本阶段需生成两类离线资产：

1. 类别属性描述文件，建议路径：
   - `assets/anchors/descriptions/{dataset}.json`
2. CLIP 强文本锚点张量文件，建议路径：
   - `assets/anchors/features/{dataset}_{arch}.pt`

建议 JSON 结构：

```json
{
  "dataset": "Aircraft",
  "classes": {
    "Boeing 737-300": [
      "A Boeing 737-300 is visually distinguished by ...",
      "Key localized features include ..."
    ]
  },
  "meta": {
    "generator": "manual_or_llm",
    "prompt_version": "v1",
    "created_at": "YYYY-MM-DD"
  }
}
```

建议 `.pt` 结构：

```python
{
  "dataset": str,
  "arch": str,
  "classnames": List[str],
  "anchors": Tensor[K, D],
  "description_count": Dict[str, int],
  "normalization": "l2",
  "prompt_version": str
}
```

## 4. 功能需求

### R1. 类别集合读取

- 系统必须能够从当前项目已有类别定义中读取类别名。
- 对于 `Aircraft`，类别顺序必须与 `data/fewshot_datasets.py` 的 `Aircraft.cname` 或对应 split 标签顺序一致。
- 对于 ImageNet 派生数据集，必须考虑 `imagenet_a_mask`、`imagenet_r_mask`、`imagenet_v_mask` 的类别子集映射，避免锚点类别顺序与 logits 列顺序错位。

### R2. 细粒度描述生成

每个类别至少生成 `M >= 3` 条细粒度视觉描述。描述应聚焦：

- 局部可见部件；
- 颜色、纹理、形状、比例；
- 与相近类别的可区分特征；
- 避免包含不可视觉观察的信息，例如产地、历史、用途、价格。

推荐生成提示：

```text
Describe the distinguishing visual characteristics of a [CLASS], focusing heavily on specific localized parts. Only mention visible cues useful for fine-grained image classification.
```

### R3. 离线缓存与可复现

- 若描述文件已存在，默认复用，不应重复请求 LLM。
- 支持强制刷新描述缓存。
- 每次生成需记录 prompt 版本、生成模型名称或人工来源、时间戳。
- 本项目网络受限，实际实现时应允许先使用手工 JSON 或本地已有描述文件，不强依赖在线 API。

### R4. 文本特征编码

- 使用冻结 CLIP 文本编码器编码每条描述。
- 对同一类别的多条描述特征求平均后执行 L2 归一化，得到：

\[
\mathbf{t}^{anchor}_k = \mathrm{Norm}\left(\frac{1}{M}\sum_{m=1}^{M}\mathcal{F}_T(d_{k,m})\right)
\]

- 输出 `anchors` 维度必须为 `[K, D]`。
- `K` 必须等于当前数据集实际分类类别数。

### R5. 与现有 PromptLearner 解耦

- 强文本锚点用于语义定位和过滤，不替代 `PromptLearner` 的可学习提示。
- 阶段一不改变 `clip/custom_clip.py` 中现有 `PromptLearner.forward()` 的行为。

## 5. 非功能需求

1. **零测试时延迟**：LLM 生成和文本编码必须在测试前完成。
2. **确定性**：相同描述文件、相同 CLIP 架构应生成一致的锚点张量。
3. **可检查性**：描述 JSON 需便于人工审阅与修改。
4. **低耦合**：新增资产生成逻辑应独立于 `tpt_classification.py` 主评估脚本。
5. **设备兼容**：支持 CPU 生成锚点；GPU 仅作为加速选项。

## 6. 建议涉及文件

> 本文档仅定义需求，不要求立即编码。

未来实现时建议新增：

- `scripts/build_text_anchors.py`
- `assets/anchors/descriptions/`
- `assets/anchors/features/`
- `docs/stage1_requirements.md`

可能复用或读取：

- `clip/clip.py`
- `clip/custom_clip.py`
- `data/cls_to_names.py`
- `data/imagnet_prompts.py`
- `data/imagenet_variants.py`

## 7. 验收标准

1. 给定一个数据集和 CLIP 架构，能够生成对应 JSON 与 `.pt` 锚点文件。
2. `.pt` 文件中的 `classnames` 顺序与评估时 logits 类别顺序一致。
3. `anchors` 每一行 L2 范数接近 1。
4. 对任意类别，至少存在 3 条聚焦局部视觉属性的描述。
5. 重复运行时可复用缓存，不破坏已有结果。

## 8. 第一阶段实现与验收记录

本阶段已提供离线锚点资产构建能力，代码入口如下：

- `utils/text_anchors.py`
  - 数据集别名规范化；
  - 按当前 TPT 评估顺序解析类别名；
  - 生成离线 fallback 视觉描述；
  - 校验 description JSON；
  - 加载并校验 encoded anchor `.pt`。
- `scripts/build_text_anchors.py`
  - 创建或复用描述 JSON；
  - 使用冻结 CLIP 文本编码器编码描述；
  - 对同类多描述向量求平均并 L2 归一化；
  - 保存强文本锚点 `.pt`。
- `scripts/validate_text_anchors.py`
  - 验收 description JSON 与可选 `.pt` 锚点文件。
- `tests/test_stage1_text_anchors.py`
  - 纯 Python 单元测试，不加载 CLIP，不依赖数据集文件。

### 8.1 生成描述 JSON，不编码 CLIP

适用于先人工检查或替换 LLM 描述：

```bash
python scripts/build_text_anchors.py \
  --dataset Pets \
  --arch ViT-B/16 \
  --skip-encode
```

默认生成：

- `assets/anchors/descriptions/Pets.json`

说明：当前实现不在线调用 LLM。若需要使用 GPT/Llama 等生成的细粒度描述，直接编辑或替换该 JSON，再重新运行编码步骤即可。

### 8.2 编码强文本锚点

在具备项目依赖、CLIP 权重和运行设备的环境中执行：

```bash
python scripts/build_text_anchors.py \
  --dataset Pets \
  --arch ViT-B/16 \
  --description-path assets/anchors/descriptions/Pets.json \
  --output assets/anchors/features/Pets_ViT-B-16.pt \
  --device cuda
```

输出 `.pt` 包含：

- `dataset`
- `arch`
- `classnames`
- `anchors: Tensor[K, D]`
- `description_count`
- `normalization: l2`
- `description_sha256`
- `created_at`
- `meta`

### 8.3 阶段验收命令

仅校验描述 JSON：

```bash
python scripts/validate_text_anchors.py \
  --dataset Pets \
  --description-path assets/anchors/descriptions/Pets.json
```

校验描述 JSON 与锚点 `.pt`，并检查 L2 范数：

```bash
python scripts/validate_text_anchors.py \
  --dataset Pets \
  --description-path assets/anchors/descriptions/Pets.json \
  --anchor-path assets/anchors/features/Pets_ViT-B-16.pt \
  --check-norm
```

运行纯工具测试：

```bash
python -m unittest tests/test_stage1_text_anchors.py
```

### 8.4 当前阶段边界

- 已完成阶段一的资产生成与验收脚本。
- 未集成到 `tpt_classification.py`，该工作属于阶段二至阶段四。
- 未在本地运行代码或配置环境；上述命令供具备依赖的运行环境验收使用。

### 8.5 使用 OpenAI-compatible 接口生成 LLM 描述

已补充 `--llm-generate` 模式。该模式会调用兼容 OpenAI Chat Completions 协议的接口，为每个类别生成细粒度视觉描述，并写入 `--description-path` 指定的 JSON 缓存。

示例：使用官方 OpenAI 或其他兼容服务：

```bash
export OPENAI_API_KEY="你的 API Key"

python scripts/build_text_anchors.py \
  --dataset Pets \
  --arch ViT-B/16 \
  --llm-generate \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini \
  --description-path assets/anchors/descriptions/Pets.json \
  --skip-encode
```

如果是第三方 OpenAI-compatible 服务，只需替换：

```bash
python scripts/build_text_anchors.py \
  --dataset Cars \
  --llm-generate \
  --llm-base-url https://your-provider.example.com/v1 \
  --llm-model your-model-name \
  --llm-api-key-env YOUR_PROVIDER_API_KEY \
  --skip-encode
```

说明：

- `--llm-base-url` 可以是 `https://host/v1`，脚本会自动拼成 `/chat/completions`。
- 也可以直接传完整 endpoint：`https://host/v1/chat/completions`。
- 默认从 `OPENAI_API_KEY` 读取 key，也可以用 `--llm-api-key-env` 指定其他环境变量。
- 不建议直接使用 `--llm-api-key`，避免 key 留在 shell history。
- 已存在且描述数量满足 `--min-descriptions` 的类别会被跳过，便于断点续跑。
- 若要重新生成全部类别，使用 `--force-description`。
- 生成描述后，可去掉 `--skip-encode` 继续编码 CLIP anchors。

完整生成并编码示例：

```bash
export OPENAI_API_KEY="你的 API Key"

python scripts/build_text_anchors.py \
  --dataset Aircraft \
  --arch ViT-B/16 \
  --llm-generate \
  --llm-model gpt-4o-mini \
  --description-path assets/anchors/descriptions/Aircraft.json \
  --output assets/anchors/features/Aircraft_ViT-B-16.pt \
  --device cuda
```
