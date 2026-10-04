# Transformers 文档 RAG 基线评估报告

任务：`e2a170d05d8f4050bc8d13e682c9407d`。所有统计来自此任务的原始磁盘记录；未补跑、未调参、未执行 Load Test。

## 结果与完整性

三种模式均生成了60道回答，semantic、keyword、distractor各20题。720个评分项全部尝试，716项成功、4项失败；每种模式59题四项评分齐全，记录状态均为 PARTIAL。任务 SUCCESS 只代表后台流程结束，不代表所有评分成功。

| 模式 | 回答完成 | 四项评分齐全的题数 | 有效评分 | 失败评分 | 未尝试评分 |
|---|---:|---:|---:|---:|---:|
| vector_only | 60/60 | 59/60 | 239/240 | 1 | 0 |
| hybrid | 60/60 | 59/60 | 238/240 | 2 | 0 |
| hybrid_rerank | 60/60 | 59/60 | 239/240 | 1 | 0 |

## 整体及分组评分

单元格为均分（有效题数／该组总题数）；失败分数不计作0。不同有效样本数量的均分不能完全直接比较。四项指标分别呈现，不合成为一个“总分”。

| 模式 | 分组 | Context Recall | Context Precision | Faithfulness | Answer Correctness |
|---|---|---:|---:|---:|---:|
| vector_only | 整体 | 0.8203 (59/60) | 0.8896 (60/60) | 0.6801 (60/60) | 0.5373 (60/60) |
| vector_only | semantic | 0.8430 (19/20) | 0.9069 (20/20) | 0.7035 (20/20) | 0.5376 (20/20) |
| vector_only | keyword | 0.8775 (20/20) | 0.8667 (20/20) | 0.6681 (20/20) | 0.5234 (20/20) |
| vector_only | distractor | 0.7416 (20/20) | 0.8952 (20/20) | 0.6686 (20/20) | 0.5510 (20/20) |
| hybrid | 整体 | 0.8342 (60/60) | 0.8987 (60/60) | 0.7043 (59/60) | 0.5629 (59/60) |
| hybrid | semantic | 0.7890 (20/20) | 0.9215 (20/20) | 0.6945 (19/20) | 0.5864 (19/20) |
| hybrid | keyword | 0.8917 (20/20) | 0.8860 (20/20) | 0.7277 (20/20) | 0.5725 (20/20) |
| hybrid | distractor | 0.8218 (20/20) | 0.8885 (20/20) | 0.6902 (20/20) | 0.5309 (20/20) |
| hybrid_rerank | 整体 | 0.8270 (59/60) | 0.9073 (60/60) | 0.7483 (60/60) | 0.5716 (60/60) |
| hybrid_rerank | semantic | 0.8115 (20/20) | 0.9255 (20/20) | 0.7792 (20/20) | 0.5900 (20/20) |
| hybrid_rerank | keyword | 0.8850 (19/20) | 0.9173 (20/20) | 0.7087 (20/20) | 0.5571 (20/20) |
| hybrid_rerank | distractor | 0.7875 (20/20) | 0.8791 (20/20) | 0.7569 (20/20) | 0.5677 (20/20) |

### 相同有效题目的配对比较

每个指标仅使用三种模式均成功评分的共同题目，减少缺失样本带来的比较偏差。

| 指标 | 共同题数 | vector_only | hybrid | hybrid_rerank |
|---|---:|---:|---:|---:|
| context_recall | 58 | 0.8172 | 0.8371 | 0.8284 |
| context_precision | 60 | 0.8896 | 0.8987 | 0.9073 |
| faithfulness | 59 | 0.6799 | 0.7043 | 0.7440 |
| answer_correctness | 59 | 0.5391 | 0.5629 | 0.5727 |

## 耗时口径

| 模式 | 本轮评估总耗时（分钟） | 每题平均评估耗时（秒） | 平均检索耗时（毫秒） | 检索耗时中位数／P95（毫秒） |
|---|---:|---:|---:|---:|
| vector_only | 58.64 | 58.64 | 69.72 | 68.27／87.53 |
| hybrid | 71.66 | 71.66 | 73.31 | 71.86／79.29 |
| hybrid_rerank | 75.04 | 75.04 | 15959.95 | 16876.58／18663.65 |

任务从 2026-10-03 21:39:27 到 2026-10-04 01:04:48（America/New_York），约 205.35 分钟。

总评估耗时包含回答生成、RAGAS多次模型调用、失败重试等待及检查点写入；不等于用户问答延迟。平均检索耗时来自每题保存的 retrieval_latency_ms，可能包含模型加载、重排序和缓存状态影响。中位数取中间两值均值，P95采用nearest-rank；这些是单任务顺序检索样本，不是并发Load Test的P99。每题生成耗时、每项评分耗时未单独保存，无法准确拆分。

## 失败与限制

| 模式 | 题目 | 指标 | 已记录错误 |
|---|---|---|---|
| vector_only | semantic_017 | context_recall | OutputParserException: scoring failed; resume to retry this metric |
| hybrid | semantic_017 | faithfulness | TimeoutError: scoring failed; resume to retry this metric |
| hybrid | semantic_017 | answer_correctness | TimeoutError: scoring failed; resume to retry this metric |
| hybrid_rerank | keyword_012 | context_recall | RagasOutputParserException: scoring failed; resume to retry this metric |

这些记录只保留异常类型，未保存评分模型的原始判断、失败响应或逐声明判定。因此能确认解析失败／超时，无法仅凭记录确定解析错误的具体内容。没有自动重试这些4项。

生成模型和评分模型均为本地 llama3.1:8b，评分存在模型偏差；只有一次运行，不能声称统计显著。60题同时用于分析和后续调参会造成测试集适配，应另留未参与调参的验证题。

## 实验配置

同一数据集SHA256：`91d4adebe3e4d2d9ae21e95f6514c9fc3110003dde8df4abc1ab49bb221ec8c5`；同一Milvus集合：`rag_0ad5fb6330c120d86100faef_54191245`。

BGE-m3向量检索，Top-K=50，HNSW ef=64；hybrid使用weighted_sum，向量／BM25权重0.7／0.3。重排序模型bge-reranker-base，ONNX启用，top_n=20、batch_size=16。回答最终使用5个上下文片段，最多16000字符。生成temperature=0、seed=42、max_tokens=512。参数来自各次不可变快照，未改当前YAML。

## 典型低分题与证据

以下原文直接取自结果记录；具体分析与完整原文分别展示，避免把低评分直接当作确定的失败原因。

### semantic_017

观察：问题针对长对话的KV缓存。vector_only和hybrid回答混入梯度累积、GPTQ等训练或权重量化内容；rerank把kv_cache.md的当前层GPU缓存／下一层预取证据排到第一，但回答仍扩展出7个问题。rerank答案正确性0.5075、忠实度1.000，说明“有证据”不等于“针对问题”。推测：更聚焦的上下文和明确只回答用户所问的提示词可能减少偏题；需小实验确认。

#### vector_only

**Question**

Long conversations are exhausting GPU memory because their accumulated attention state is large. How can we keep less of that state on the GPU, and what performance trade-off follows?

**Reference**

Use cache offloading. Most layers’ KV caches are moved to CPU, while the current layer’s cache is maintained on GPU and the next layer is prefetched. This saves GPU memory, but transfers between CPU and GPU can reduce generation throughput.

**实际回答**

Based on the provided evidence, here are the answers to your questions:

1. How can we keep less of the attention state on the GPU?

According to [1], you can pass the `max_memory` parameter to allocate a specific amount of memory on the GPU. However, this does not directly address the attention state.

According to [2], gradient accumulation can reduce memory usage by storing fewer gradients. This can also enable training with a larger effective batch size.

According to [4], cache offloading can save GPU memory by moving the KV cache for model layers except one to the CPU.

2. What performance trade-off follows from these methods?

According to [2], gradient accumulation can slow down training due to the additional forward and backward passes introduced.

According to [4], cache offloading can trade off speed for reduced memory usage.

According to [5], using Adafactor or an 8-bit AdamW optimizer can reduce memory requirements, but at the cost of slower convergence.

Note that the available documentation does not provide a direct answer to how to keep less of the attention state on the GPU. However, the methods mentioned above can help reduce memory usage and potentially alleviate the issue of exhausting GPU memory.

**评分**

context_recall=None, context_precision=0.4777777777618519, faithfulness=0.6923076923076923, answer_correctness=0.4336183582972993

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_11a690e9d8c3d241`

```text
If you're running out of memory because a dataset is too large (disk offloading is not supported), try passing the `max_memory` parameter to allocate the amount of memory to use on your device (GPU and CPU).

```py
quantized_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",
    max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"},
    quantization_config=gptq_config
)
```
```

证据[2]：`transformers-v4.57.1/perf_train_gpu_one.md`；chunk_id=`79ad94e6e025a820048a1661_46a3f16c767c3326`

```text
### Gradient accumulation

Gradient accumulation overcomes memory constraints - useful for fitting a very large model that otherwise wouldn't fit on a single GPU - by accumulating gradients over multiple mini-batches before updating the parameters.

This reduces memory by storing fewer gradients and enables training with a larger *effective batch size* because usually, the parameters are updated from a single batch of data.
```

证据[3]：`transformers-v4.57.1/perf_train_gpu_one.md`；chunk_id=`79ad94e6e025a820048a1661_cf374c277754ffc9`

```text
Training can slow down though due to the additional forward and backward passes introduced by gradient accumulation.

Configure [`~TrainingArguments.per_device_train_batch_size`] in [`TrainingArguments`] to enable gradient accumulation.

```py
from transformers import TrainingArguments

# effective batch size of 64
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
)
```
```

证据[4]：`transformers-v4.57.1/kv_cache.md`；chunk_id=`e49d5e88f644688597f01749_aa8f73f7c218a4d4`

```text
The KV cache can occupy a significant portion of memory and become a [bottleneck](https://hf.co/blog/llama31#inference-memory-requirements) for long-context generation.

Memory efficient caches focus on trading off speed for reduced memory usage.

This is especially important for large language models (LLMs) and if your hardware is memory constrained.

Offloading the cache saves GPU memory by moving the KV cache for model layers except one to the CPU.
```

证据[5]：`transformers-v4.57.1/perf_train_gpu_one.md`；chunk_id=`79ad94e6e025a820048a1661_c0b321f986eab4a0`

```text
For example, Adafactor can significantly reduce memory requirements by storing a weighted average of a row or column instead of each element in the matrix at the cost of slower convergence.

Another example is using a [8-bit AdamW optimizer](https://huggingface.co/docs/bitsandbytes) from bitsandbytes to quantize optimizer states.

The optimizer state is stored in a lower precision and dequantized before being used in the optimizer step.
```


#### hybrid

**Question**

Long conversations are exhausting GPU memory because their accumulated attention state is large. How can we keep less of that state on the GPU, and what performance trade-off follows?

**Reference**

Use cache offloading. Most layers’ KV caches are moved to CPU, while the current layer’s cache is maintained on GPU and the next layer is prefetched. This saves GPU memory, but transfers between CPU and GPU can reduce generation throughput.

**实际回答**

Based on the provided evidence, here are the answers to your questions:

1. How can we keep less of the attention state on the GPU?

According to [1], gradient accumulation can reduce memory usage by storing fewer gradients. This can be achieved by accumulating gradients over multiple mini-batches before updating the parameters.

2. What performance trade-off follows?

According to [5], training can slow down due to the additional forward and backward passes introduced by gradient accumulation.

3. How can we offload the cache to save GPU memory?

According to [2], the KV cache can be offloaded to the CPU by moving the KV cache for model layers except one to the CPU. This can save GPU memory by reducing the memory occupied by the KV cache.

4. What is the memory usage reduction achieved by GPTQ?

According to [4], GPTQ can save memory usage by 4x due to the dequantization of int4 weights in a fused kernel rather than a GPU's global memory.

5. What is the performance impact of GPTQ on inference?

According to [4], inference is also faster with GPTQ because a lower bitwidth takes less time to communicate.

**评分**

context_recall=0.5, context_precision=0.4499999999775, faithfulness=None, answer_correctness=None

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/perf_train_gpu_one.md`；chunk_id=`79ad94e6e025a820048a1661_46a3f16c767c3326`

```text
### Gradient accumulation

Gradient accumulation overcomes memory constraints - useful for fitting a very large model that otherwise wouldn't fit on a single GPU - by accumulating gradients over multiple mini-batches before updating the parameters.

This reduces memory by storing fewer gradients and enables training with a larger *effective batch size* because usually, the parameters are updated from a single batch of data.
```

证据[2]：`transformers-v4.57.1/kv_cache.md`；chunk_id=`e49d5e88f644688597f01749_aa8f73f7c218a4d4`

```text
The KV cache can occupy a significant portion of memory and become a [bottleneck](https://hf.co/blog/llama31#inference-memory-requirements) for long-context generation.

Memory efficient caches focus on trading off speed for reduced memory usage.

This is especially important for large language models (LLMs) and if your hardware is memory constrained.

Offloading the cache saves GPU memory by moving the KV cache for model layers except one to the CPU.
```

证据[3]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_11a690e9d8c3d241`

```text
If you're running out of memory because a dataset is too large (disk offloading is not supported), try passing the `max_memory` parameter to allocate the amount of memory to use on your device (GPU and CPU).

```py
quantized_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",
    max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"},
    quantization_config=gptq_config
)
```
```

证据[4]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_9d22d7ab1de86eb9`

```text
This can save memory usage by 4x because the int4 weights are dequantized in a fused kernel rather than a GPU's global memory.

Inference is also faster because a lower bitwidth takes less time to communicate.

> [!WARNING]
> AutoGPTQ is likely to be deprecated in the future due to lack of continued support for new models and features.
```

证据[5]：`transformers-v4.57.1/perf_train_gpu_one.md`；chunk_id=`79ad94e6e025a820048a1661_cf374c277754ffc9`

```text
Training can slow down though due to the additional forward and backward passes introduced by gradient accumulation.

Configure [`~TrainingArguments.per_device_train_batch_size`] in [`TrainingArguments`] to enable gradient accumulation.

```py
from transformers import TrainingArguments

# effective batch size of 64
args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
)
```
```


#### hybrid_rerank

**Question**

Long conversations are exhausting GPU memory because their accumulated attention state is large. How can we keep less of that state on the GPU, and what performance trade-off follows?

**Reference**

Use cache offloading. Most layers’ KV caches are moved to CPU, while the current layer’s cache is maintained on GPU and the next layer is prefetched. This saves GPU memory, but transfers between CPU and GPU can reduce generation throughput.

**实际回答**

Based on the provided evidence, here are the answers to your questions:

1. How can we keep less of the attention state on the GPU?

According to [1], you can consider offloading the cache to the CPU, which will save GPU memory by moving the KV cache for model layers except one to the CPU.

2. What performance trade-off follows from offloading the cache?

According to [5], offloading the cache trades off speed for reduced memory usage. This is especially important for large language models (LLMs) and if your hardware is memory constrained.

3. What other options are available to reduce GPU memory usage?

According to [2], you can enable `offload_optimizer` to reduce GPU memory usage. Additionally, you can configure `overlap_comm` to use more GPU memory in exchange for lower allreduce latency, but be careful not to exceed your GPU memory limit.

4. How can we balance available GPU memory and communication speed?

According to [2], you can adjust the values of `allgather_bucket_size` and `reduce_bucket_size` to balance available GPU memory and communication speed. Smaller values will make communication slower but use more GPU memory.

5. Can we offload gradients to the CPU to reduce GPU memory usage?

According to [2], `round_robin_gradients` is available in DeepSpeed 0.4.4 for CPU offloading, which parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning.

6. Can we limit the amount of memory used on the GPU and CPU?

According to [3], you can pass the `max_memory` parameter to allocate the amount of memory to use on your device (GPU and CPU).

7. Can we accumulate gradients over multiple mini-batches to reduce memory usage?

According to [4], gradient accumulation overcomes memory constraints by accumulating gradients over multiple mini-batches before updating the parameters, which reduces memory usage and enables training with a larger effective batch size.

**评分**

context_recall=0.75, context_precision=0.8666666666377778, faithfulness=1.0, answer_correctness=0.5074639148178418

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/kv_cache.md`；chunk_id=`e49d5e88f644688597f01749_fc26fc913947530a`

```text
Only the current layer cache is maintained on the GPU during a models `forward` iteration over the layers.

It will asynchronously prefetch the next layer's cache, and send back the current layer's cache back to the CPU after attention computation.

You may want to consider offloading if you have a small GPU and you're getting out-of-memory (OOM) errors.
```

证据[2]：`transformers-v4.57.1/deepspeed.md`；chunk_id=`afeccf9809f36585a19d9172_1b73ceb189d42568`

```text
```

</hfoption>
<hfoption id="ZeRO-2">

ZeRO-2 shards the optimizer and gradient states across GPUs. This stage is primarily used for training since its features are not relevant to inference. Some important parameters to configure for better performance include the following.

* `offload_optimizer` should be enabled to reduce GPU memory usage.
* `overlap_comm` when set to `true` uses more GPU memory in exchange for lower allreduce latency. This feature uses 4.5x the `allgather_bucket_size` and `reduce_bucket_size` values. In this example, they're set to `5e8` which means it requires 9GB of GPU memory. If your GPU memory is 8GB or less, you should reduce `overlap_comm` to lower the memory requirements and prevent an out-of-memory (OOM) error.
* `allgather_bucket_size` and `reduce_bucket_size` trade-off available GPU memory for communication speed. The smaller their values, the slower communication is and the more GPU memory is available. You can balance, for example, whether a bigger batch size is more important than a slightly slower training time.
* `round_robin_gradients` is available in DeepSpeed 0.4.4 for CPU offloading. It parallelizes gradient copying to CPU memory among ranks by fine-grained gradient partitioning. Performance benefit grows with gradient accumulation steps (more copying between optimizer steps) or GPU count (increased parallelism).

```yml
```

证据[3]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_11a690e9d8c3d241`

```text
If you're running out of memory because a dataset is too large (disk offloading is not supported), try passing the `max_memory` parameter to allocate the amount of memory to use on your device (GPU and CPU).

```py
quantized_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",
    max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"},
    quantization_config=gptq_config
)
```
```

证据[4]：`transformers-v4.57.1/perf_train_gpu_one.md`；chunk_id=`79ad94e6e025a820048a1661_46a3f16c767c3326`

```text
### Gradient accumulation

Gradient accumulation overcomes memory constraints - useful for fitting a very large model that otherwise wouldn't fit on a single GPU - by accumulating gradients over multiple mini-batches before updating the parameters.

This reduces memory by storing fewer gradients and enables training with a larger *effective batch size* because usually, the parameters are updated from a single batch of data.
```

证据[5]：`transformers-v4.57.1/kv_cache.md`；chunk_id=`e49d5e88f644688597f01749_aa8f73f7c218a4d4`

```text
The KV cache can occupy a significant portion of memory and become a [bottleneck](https://hf.co/blog/llama31#inference-memory-requirements) for long-context generation.

Memory efficient caches focus on trading off speed for reduced memory usage.

This is especially important for large language models (LLMs) and if your hardware is memory constrained.

Offloading the cache saves GPU memory by moving the KV cache for model layers except one to the CPU.
```


### keyword_002

观察：三个模式均明确给出only_second，但答案正确性只有约0.35–0.37；参考答案额外包含max_length默认行为。pad_truncation.md检索片段将only_second条目和解释拆开，部分片段开头是上一条目的尾句，随后出现only_first。推测：结构破碎可能影响判分，参考答案覆盖要求也可能超过题目实际所问。应人工核对题目与参考答案，不应仅为了提高评分扩大回答。

#### vector_only

**Question**

For a pair of sequences, which truncation value preserves the first sequence and truncates only the second?

**Reference**

Use truncation="only_second". It truncates only the second sequence of a pair to satisfy the specified max_length, or the model’s maximum accepted length when max_length is not provided.

**实际回答**

Based on the provided evidence, the truncation value that preserves the first sequence and truncates only the second is `'only_second'`. This is stated in evidence [3], which describes the behavior of the `'only_second'` strategy as truncating "only the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided".

**评分**

context_recall=0.3, context_precision=0.6791666666496875, faithfulness=0.6666666666666666, answer_correctness=0.3654999844020528

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_7a68a9ed4a971d79`

```text
This will only truncate
    the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `'only_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`).
```

证据[2]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_eaf7394effb3a9ab`

```text
If you use pairs of input sequences in any of the following examples, you can replace `truncation=True` by a `STRATEGY` selected in
`['only_first', 'only_second', 'longest_first']`, i.e. `truncation='only_second'` or `truncation='longest_first'` to control how both sequences in the pair are truncated as detailed before.
```

证据[3]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_03fb26a197e46f26`

```text
This will
    truncate token by token, removing a token from the longest sequence in the pair until the proper length is
    reached.
- `'only_second'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`).
```

证据[4]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_8f251e9784851182`

```text
This will only truncate
    the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `False` or `'do_not_truncate'`: no truncation is applied.

This is the default behavior.
```

证据[5]：`transformers-v4.57.1/fast_tokenizers.md`；chunk_id=`cacc3608c636b069f5643065_2c1e3b48b1082180`

```text
> [!TIP]
> Learn about additional truncation strategies in the [Padding and truncation](./pad_truncation) guide.

Models are only able to process sequences up to a certain length.

If you try to process a sequence longer than a model can handle, it crashes.

Truncation removes tokens from a sequence to ensure it doesn't exceed the maximum length.

Set `truncation=True` to truncate a sequence to the maximum length accepted by the model.
```


#### hybrid

**Question**

For a pair of sequences, which truncation value preserves the first sequence and truncates only the second?

**Reference**

Use truncation="only_second". It truncates only the second sequence of a pair to satisfy the specified max_length, or the model’s maximum accepted length when max_length is not provided.

**实际回答**

Based on the provided evidence, the truncation value that preserves the first sequence and truncates only the second is `'only_second'`.

This is stated in evidence [4], which describes the behavior of the `'only_second'` strategy: "This will only truncate the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided."

Therefore, the answer is: `'only_second'`.

**评分**

context_recall=1.0, context_precision=0.8041666666465626, faithfulness=1.0, answer_correctness=0.35108494910188803

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_eaf7394effb3a9ab`

```text
If you use pairs of input sequences in any of the following examples, you can replace `truncation=True` by a `STRATEGY` selected in
`['only_first', 'only_second', 'longest_first']`, i.e. `truncation='only_second'` or `truncation='longest_first'` to control how both sequences in the pair are truncated as detailed before.
```

证据[2]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_7a68a9ed4a971d79`

```text
This will only truncate
    the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `'only_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`).
```

证据[3]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_8f251e9784851182`

```text
This will only truncate
    the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `False` or `'do_not_truncate'`: no truncation is applied.

This is the default behavior.
```

证据[4]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_03fb26a197e46f26`

```text
This will
    truncate token by token, removing a token from the longest sequence in the pair until the proper length is
    reached.
- `'only_second'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`).
```

证据[5]：`transformers-v4.57.1/fast_tokenizers.md`；chunk_id=`cacc3608c636b069f5643065_2c1e3b48b1082180`

```text
> [!TIP]
> Learn about additional truncation strategies in the [Padding and truncation](./pad_truncation) guide.

Models are only able to process sequences up to a certain length.

If you try to process a sequence longer than a model can handle, it crashes.

Truncation removes tokens from a sequence to ensure it doesn't exceed the maximum length.

Set `truncation=True` to truncate a sequence to the maximum length accepted by the model.
```


#### hybrid_rerank

**Question**

For a pair of sequences, which truncation value preserves the first sequence and truncates only the second?

**Reference**

Use truncation="only_second". It truncates only the second sequence of a pair to satisfy the specified max_length, or the model’s maximum accepted length when max_length is not provided.

**实际回答**

Based on the provided evidence, the truncation value that preserves the first sequence and truncates only the second is `'only_second'`. This is stated in evidence [3], which describes the behavior of the `'only_second'` strategy as truncating "only the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided".

**评分**

context_recall=0.3, context_precision=0.6791666666496875, faithfulness=0.6666666666666666, answer_correctness=0.3654999844020528

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_7a68a9ed4a971d79`

```text
This will only truncate
    the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `'only_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`).
```

证据[2]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_eaf7394effb3a9ab`

```text
If you use pairs of input sequences in any of the following examples, you can replace `truncation=True` by a `STRATEGY` selected in
`['only_first', 'only_second', 'longest_first']`, i.e. `truncation='only_second'` or `truncation='longest_first'` to control how both sequences in the pair are truncated as detailed before.
```

证据[3]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_03fb26a197e46f26`

```text
This will
    truncate token by token, removing a token from the longest sequence in the pair until the proper length is
    reached.
- `'only_second'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`).
```

证据[4]：`transformers-v4.57.1/pad_truncation.md`；chunk_id=`96b3fa6d18bf59442ac4ae89_8f251e9784851182`

```text
This will only truncate
    the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `False` or `'do_not_truncate'`: no truncation is applied.

This is the default behavior.
```

证据[5]：`transformers-v4.57.1/fast_tokenizers.md`；chunk_id=`cacc3608c636b069f5643065_2c1e3b48b1082180`

```text
> [!TIP]
> Learn about additional truncation strategies in the [Padding and truncation](./pad_truncation) guide.

Models are only able to process sequences up to a certain length.

If you try to process a sequence longer than a model can handle, it crashes.

Truncation removes tokens from a sequence to ensure it doesn't exceed the maximum length.

Set `truncation=True` to truncate a sequence to the maximum length accepted by the model.
```


### distractor_006

观察：问题区分“按配置构造”与“加载预训练权重”。vector_only已检索到models.md的random weights说明，却用训练分类头未初始化和raw hidden states作为论据；rerank回答直接把按配置构造描述为from_pretrained，并声称证据不足。可见实际概念混淆；这道题需要更清晰的构造API上下文和针对性回答约束，单纯rerank未解决。

#### vector_only

**Question**

Does constructing a model from its configuration give us the pretrained checkpoint’s learned weights?

**Reference**

No. A configuration describes the architecture and its attributes; constructing the model this way yields random weights that need training. Load learned checkpoint weights with from_pretrained instead.

**实际回答**

Based on the provided evidence, constructing a model from its configuration does not necessarily give us the pre-trained checkpoint's learned weights.

[4] Document c45b234ccb1f8c... shows an example where the model is loaded from a pre-trained checkpoint, but some weights are not initialized from the checkpoint and are newly initialized. This suggests that the pre-trained weights are not automatically loaded when constructing a model from its configuration.

Additionally, [2] Document a3440bbaccf7c... states that a pre-trained model only returns the raw hidden states, implying that the pre-trained weights are not directly accessible.

Therefore, it appears that constructing a model from its configuration does not give us the pre-trained checkpoint's learned weights.

**评分**

context_recall=0.8, context_precision=0.249999999975, faithfulness=1.0, answer_correctness=0.34705960599801106

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_61cf23d239b1f965`

```text
At this point, you have a model with random weights that needs to be trained to output meaningful results.

> [!TIP]
> An *architecture* refers to the model's skeleton and a *checkpoint* refers to the model's weights for a given architecture.

For example, [BERT](./model_doc/bert) is an architecture while [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) is a checkpoint.

You'll see the term *model* used interchangeably with architecture and checkpoint.
```

证据[2]：`transformers-v4.57.1/quicktour.md`；chunk_id=`a3440bbaccf7c717b0577b0b_79a023b63c1e399d`

```text
Each pretrained model inherits from three base classes.

| **Class** | **Description** |
|---|---|
| [`PretrainedConfig`] | A file that specifies a models attributes such as the number of attention heads or vocabulary size. |
| [`PreTrainedModel`] | A model (or architecture) defined by the model attributes from the configuration file.

A pretrained model only returns the raw hidden states.
```

证据[3]：`transformers-v4.57.1/quicktour.md`；chunk_id=`a3440bbaccf7c717b0577b0b_8b05079fb6e61565`

```text
We recommend using the [AutoClass](./model_doc/auto) API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use [`~PreTrainedModel.from_pretrained`] to load the weights and configuration file from the Hub into the model and preprocessor class.
```

证据[4]：`transformers-v4.57.1/training.md`；chunk_id=`c45b234ccb1f8cb4d2c087f3_9f5ab1dc0066582d`

```text
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
"Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']"
"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
```

证据[5]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_2a2ea103381ac095`

```text
```py
from transformers.modeling_utils import load_sharded_checkpoint

with tempfile.TemporaryDirectory() as tmp_dir:
    model.save_pretrained(tmp_dir, max_shard_size="5GB")
    load_sharded_checkpoint(model, tmp_dir)
```

The [`~PreTrainedModel.save_pretrained`] method creates an index file that maps parameter names to the files they're stored in.

The index file has two keys, `metadata` and `weight_map`.
```


#### hybrid

**Question**

Does constructing a model from its configuration give us the pretrained checkpoint’s learned weights?

**Reference**

No. A configuration describes the architecture and its attributes; constructing the model this way yields random weights that need training. Load learned checkpoint weights with from_pretrained instead.

**实际回答**

Based on the provided evidence, constructing a model from its configuration does not give us the pre-trained checkpoint's learned weights.

[4] states that a pretrained model inherits from `PreTrainedModel`, which is a model (or architecture) defined by the model attributes from the configuration file. This suggests that the configuration file only defines the model's architecture, not its weights.

Additionally, [5] shows an example of fine-tuning a pre-trained model, where the model is loaded using `AutoModelForSequenceClassification.from_pretrained`, and the output message indicates that some weights were not initialized from the model checkpoint and are newly initialized. This implies that the pre-trained weights are not loaded by default when constructing a model from its configuration.

Therefore, the available documentation is insufficient to confirm whether constructing a model from its configuration gives us the pre-trained checkpoint's learned weights.

**评分**

context_recall=1.0, context_precision=0.19999999998, faithfulness=1.0, answer_correctness=0.37406382518245257

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_61cf23d239b1f965`

```text
At this point, you have a model with random weights that needs to be trained to output meaningful results.

> [!TIP]
> An *architecture* refers to the model's skeleton and a *checkpoint* refers to the model's weights for a given architecture.

For example, [BERT](./model_doc/bert) is an architecture while [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) is a checkpoint.

You'll see the term *model* used interchangeably with architecture and checkpoint.
```

证据[2]：`transformers-v4.57.1/quicktour.md`；chunk_id=`a3440bbaccf7c717b0577b0b_8b05079fb6e61565`

```text
We recommend using the [AutoClass](./model_doc/auto) API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use [`~PreTrainedModel.from_pretrained`] to load the weights and configuration file from the Hub into the model and preprocessor class.
```

证据[3]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_4f3a4ef28bcebf49`

```text
# Loading models

Transformers provides many pretrained models that are ready to use with a single line of code.

It requires a model class and the [`~PreTrainedModel.from_pretrained`] method.

Call [`~PreTrainedModel.from_pretrained`] to download and load a model's weights and configuration stored on the Hugging Face [Hub](https://hf.co/models).
```

证据[4]：`transformers-v4.57.1/quicktour.md`；chunk_id=`a3440bbaccf7c717b0577b0b_79a023b63c1e399d`

```text
Each pretrained model inherits from three base classes.

| **Class** | **Description** |
|---|---|
| [`PretrainedConfig`] | A file that specifies a models attributes such as the number of attention heads or vocabulary size. |
| [`PreTrainedModel`] | A model (or architecture) defined by the model attributes from the configuration file.

A pretrained model only returns the raw hidden states.
```

证据[5]：`transformers-v4.57.1/training.md`；chunk_id=`c45b234ccb1f8cb4d2c087f3_9f5ab1dc0066582d`

```text
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
"Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']"
"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
```


#### hybrid_rerank

**Question**

Does constructing a model from its configuration give us the pretrained checkpoint’s learned weights?

**Reference**

No. A configuration describes the architecture and its attributes; constructing the model this way yields random weights that need training. Load learned checkpoint weights with from_pretrained instead.

**实际回答**

Based on the provided evidence, constructing a model from its configuration using the `~PreTrainedModel.from_pretrained` method does not necessarily give us the pre-trained checkpoint's learned weights.

[4] Document c45b234ccb1f8cb4d2c087f3, section Fine-tuning, shows that when loading a pre-trained model, some weights may not be initialized from the model checkpoint and are newly initialized instead. This suggests that the weights loaded by `~PreTrainedModel.from_pretrained` may not be the exact same weights as the pre-trained checkpoint.

Additionally, [5] Document 7b000f4c9f028a4eb28c2bcf, section Large models, explains the loading process, which involves creating a model with random weights, loading the pre-trained weights, and then placing the pre-trained weights on the model. This implies that the model starts with random weights and then the pre-trained weights are loaded, rather than the pre-trained weights being used directly.

Therefore, the available documentation is insufficient to confirm that constructing a model from its configuration gives us the pre-trained checkpoint's learned weights.

**评分**

context_recall=0.5, context_precision=0.32499999998375, faithfulness=0.5, answer_correctness=0.35539662851534126

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/quicktour.md`；chunk_id=`a3440bbaccf7c717b0577b0b_8b05079fb6e61565`

```text
We recommend using the [AutoClass](./model_doc/auto) API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use [`~PreTrainedModel.from_pretrained`] to load the weights and configuration file from the Hub into the model and preprocessor class.
```

证据[2]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_4f3a4ef28bcebf49`

```text
# Loading models

Transformers provides many pretrained models that are ready to use with a single line of code.

It requires a model class and the [`~PreTrainedModel.from_pretrained`] method.

Call [`~PreTrainedModel.from_pretrained`] to download and load a model's weights and configuration stored on the Hugging Face [Hub](https://hf.co/models).
```

证据[3]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_0f1142d5af82a7c3`

```text
To get a pretrained model, you need to load the weights into the model.

This is done by calling [`~PreTrainedModel.from_pretrained`] which accepts weights from the Hugging Face Hub or a local directory.

There are two model classes, the [AutoModel](./model_doc/auto) class and a model-specific class.
```

证据[4]：`transformers-v4.57.1/training.md`；chunk_id=`c45b234ccb1f8cb4d2c087f3_9f5ab1dc0066582d`

```text
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
"Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']"
"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
```

证据[5]：`transformers-v4.57.1/models.md`；chunk_id=`7b000f4c9f028a4eb28c2bcf_1972a2aad3fd4ea1`

```text
Large pretrained models require a lot of memory to load.

The loading process involves:

1. creating a model with random weights
2. loading the pretrained weights
3. placing the pretrained weights on the model

You need enough memory to hold two copies of the model weights (random and pretrained) which may not be possible depending on your hardware.

In distributed training environments, this is even more challenging because each process loads a pretrained model.
```


### distractor_018

观察：vector_only／hybrid主结论均否认GPTQ与AWQ属于无需校准，但faithfulness均为0、answer_correctness约0.318。实际证据中存在“AWQ/GPTQ ... need calibration”，部分引用却将GPTQ算法介绍等间接内容当作校准依据，答案也没列参考答案中的三种免校准方法。rerank检索到upfront calibration表格，faithfulness为0.8、answer_correctness为0.5683。不能由主结论正确就认定整段正确，也不能据0分断言整段没有任何依据；应人工复核逐句引用和评分偏差。

#### vector_only

**Question**

Are GPTQ and AWQ on-the-fly methods that require no calibration dataset, just like the no-calibration approaches listed in the guide?

**Reference**

No. The selecting guide groups GPTQ/GPTQModel and AWQ under Calibration-based Quantization, which requires an upfront calibration step using a dataset. bitsandbytes, HQQ and torchao are listed under No Calibration Required.

**实际回答**

Based on the provided evidence, it appears that GPTQ and AWQ are not exactly like the no-calibration approaches listed in the guide.

[2] states that GPTQ is a post-training quantization technique that requires a calibration step to find a version of the weights that minimizes the error. This suggests that GPTQ does require some form of calibration.

[3] mentions that AWQ requires calibration if quantizing yourself, which implies that it also requires some form of calibration.

However, [4] mentions that AWQ/GPTQ often lead in accuracy but need calibration, which suggests that they do require calibration, but it's worth it for the potential accuracy gains.

Therefore, the answer is that GPTQ and AWQ are not on-the-fly methods that require no calibration dataset, unlike the no-calibration approaches listed in the guide.

**评分**

context_recall=1.0, context_precision=0.8666666666377778, faithfulness=0.0, answer_correctness=0.3182545591092956

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_4e943d6ce18e00ea`

```text
### No Calibration Required (On-the-fly Quantization)

These methods are generally easier to use as they don't need a separate calibration dataset or step.
```

证据[2]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_ef96c290caf22ec5`

```text
# GPTQ

The [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) implements the GPTQ algorithm, a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes the error.

These weights are quantized to int4, but they're restored to fp16 on the fly during inference.
```

证据[3]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_319b35be93aa1101`

```text
|
| Shorter calibration time than GPTQ.                                  |                                                     |
| Many pre-quantized AWQ models on [Hugging Face Hub](https://huggingface.co/models?other=awq). |                                                     |

See the [AWQ documentation](./awq) for more details.
```

证据[4]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_eaf6953ea3010ad7`

```text
|
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4)    | ~4x                      | Relatively high accuracy            | AWQ/GPTQ often lead in accuracy but need calibration.

HQQ/bnb-nf4 are easy on-the-fly. |
| **Sub-4-bit** (VPTQ, AQLM, 2-bit GPTQ) | Extreme (>4x)            | Noticeable drop, especially at 2-bit | Quantization times can be very long (AQLM, VPTQ).

Performance varies. |
```

证据[5]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_e7480182c2c5a8c3`

```text
| Pros                                                                 | Cons                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| Often achieves high accuracy at 4-bit. (Sometimes surpasses GPTQ on specific tasks.) | Requires calibration if quantizing yourself.        |
| Can lead to inference speedups.                                      |
```


#### hybrid

**Question**

Are GPTQ and AWQ on-the-fly methods that require no calibration dataset, just like the no-calibration approaches listed in the guide?

**Reference**

No. The selecting guide groups GPTQ/GPTQModel and AWQ under Calibration-based Quantization, which requires an upfront calibration step using a dataset. bitsandbytes, HQQ and torchao are listed under No Calibration Required.

**实际回答**

Based on the provided evidence, it appears that GPTQ and AWQ are not exactly like the no-calibration approaches listed in the guide.

[2] states that GPTQ is a post-training quantization technique that requires a calibration step to find a version of the weights that minimizes the error. This suggests that GPTQ does require some form of calibration.

[3] mentions that AWQ requires calibration if quantizing yourself, which implies that it also requires some form of calibration.

However, [4] mentions that AWQ/GPTQ often lead in accuracy but need calibration, which suggests that they do require calibration, but it's worth it for the potential accuracy gains.

Therefore, the answer is that GPTQ and AWQ are not on-the-fly methods that require no calibration dataset, unlike the no-calibration approaches listed in the guide.

**评分**

context_recall=1.0, context_precision=0.8666666666377778, faithfulness=0.0, answer_correctness=0.3182545591092956

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_4e943d6ce18e00ea`

```text
### No Calibration Required (On-the-fly Quantization)

These methods are generally easier to use as they don't need a separate calibration dataset or step.
```

证据[2]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_ef96c290caf22ec5`

```text
# GPTQ

The [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) implements the GPTQ algorithm, a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes the error.

These weights are quantized to int4, but they're restored to fp16 on the fly during inference.
```

证据[3]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_319b35be93aa1101`

```text
|
| Shorter calibration time than GPTQ.                                  |                                                     |
| Many pre-quantized AWQ models on [Hugging Face Hub](https://huggingface.co/models?other=awq). |                                                     |

See the [AWQ documentation](./awq) for more details.
```

证据[4]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_eaf6953ea3010ad7`

```text
|
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4)    | ~4x                      | Relatively high accuracy            | AWQ/GPTQ often lead in accuracy but need calibration.

HQQ/bnb-nf4 are easy on-the-fly. |
| **Sub-4-bit** (VPTQ, AQLM, 2-bit GPTQ) | Extreme (>4x)            | Noticeable drop, especially at 2-bit | Quantization times can be very long (AQLM, VPTQ).

Performance varies. |
```

证据[5]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_e7480182c2c5a8c3`

```text
| Pros                                                                 | Cons                                                |
|----------------------------------------------------------------------|-----------------------------------------------------|
| Often achieves high accuracy at 4-bit. (Sometimes surpasses GPTQ on specific tasks.) | Requires calibration if quantizing yourself.        |
| Can lead to inference speedups.                                      |
```


#### hybrid_rerank

**Question**

Are GPTQ and AWQ on-the-fly methods that require no calibration dataset, just like the no-calibration approaches listed in the guide?

**Reference**

No. The selecting guide groups GPTQ/GPTQModel and AWQ under Calibration-based Quantization, which requires an upfront calibration step using a dataset. bitsandbytes, HQQ and torchao are listed under No Calibration Required.

**实际回答**

Based on the provided evidence, it appears that GPTQ and AWQ are not on-the-fly methods that require no calibration dataset, unlike the no-calibration approaches listed in the guide.

[1] Document 0f4157160ba55e76abf56443, section Benchmark Comparison states that "AWQ/GPTQ often lead in accuracy but need calibration." This suggests that GPTQ and AWQ require calibration.

[3] Document 0f4157160ba55e76abf56443, section Inference lists GPTQModel or AWQ as a quantization method that requires "upfront calibration" for good 4-bit accuracy.

[4] Document 9c43b1b786bd76932e816ae8, section GPTQ describes GPTQ as a post-training quantization technique that requires quantizing each row of the weight matrix independently to find a version of the weights that minimizes the error. This implies that GPTQ requires some form of calibration or optimization.

[5] Document 5a6d7a21b7782e6383a36f91, section AWQ describes AWQ as preserving a small fraction of the weights that are important for LLM performance to compress a model to 4-bits with minimal performance degradation. This suggests that AWQ requires some form of calibration or optimization.

Therefore, based on the available documentation, it appears that GPTQ and AWQ are not on-the-fly methods that require no calibration dataset.

**评分**

context_recall=0.5, context_precision=0.6791666666496875, faithfulness=0.8, answer_correctness=0.5683006515887266

**本次实际检索证据（完整片段）**

证据[1]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_eaf6953ea3010ad7`

```text
|
| **4-bit** (AWQ, GPTQ, HQQ, bnb-nf4)    | ~4x                      | Relatively high accuracy            | AWQ/GPTQ often lead in accuracy but need calibration.

HQQ/bnb-nf4 are easy on-the-fly. |
| **Sub-4-bit** (VPTQ, AQLM, 2-bit GPTQ) | Extreme (>4x)            | Noticeable drop, especially at 2-bit | Quantization times can be very long (AQLM, VPTQ).

Performance varies. |
```

证据[2]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_4e943d6ce18e00ea`

```text
### No Calibration Required (On-the-fly Quantization)

These methods are generally easier to use as they don't need a separate calibration dataset or step.
```

证据[3]：`transformers-v4.57.1/quantization/selecting.md`；chunk_id=`0f4157160ba55e76abf56443_6933258d4340d3e9`

```text
## Inference

Consider the quantization methods below for inference.

| quantization method | use case |
|---|---|
| bitsandbytes | ease of use and QLoRA fine-tuning on NVIDIA and Intel GPUs |
| compressed-tensors | loading specific quantized formats (FP8, Sparse) |
| GPTQModel or AWQ | good 4-bit accuracy with upfront calibration |
| HQQ | fast on the fly quantization without calibration |
| torchao | flexibility and fast inference with torch.compile |
```

证据[4]：`transformers-v4.57.1/quantization/gptq.md`；chunk_id=`9c43b1b786bd76932e816ae8_ef96c290caf22ec5`

```text
# GPTQ

The [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) implements the GPTQ algorithm, a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes the error.

These weights are quantized to int4, but they're restored to fp16 on the fly during inference.
```

证据[5]：`transformers-v4.57.1/quantization/awq.md`；chunk_id=`5a6d7a21b7782e6383a36f91_58fd1cee2f1faecf`

```text
# AWQ

[Activation-aware Weight Quantization (AWQ)](https://hf.co/papers/2306.00978) preserves a small fraction of the weights that are important for LLM performance to compress a model to 4-bits with minimal performance degradation.
```

## 结论与下一轮建议

1. 本轮hybrid_rerank整体忠实度0.7483、答案正确性0.5716最高，但平均检索约15.96秒，明显高于另外两种模式的约70毫秒。hybrid整体答案正确性0.5629，与rerank相差约0.0087；速度和质量权衡值得优先讨论，不能直接宣布rerank最优。
2. 分组表现不一致：keyword的答案正确性hybrid最高（0.5725），semantic和distractor为rerank较高。下一轮保留分组结果，避免仅优化整体均分。
3. 先人工检查上述4道案例和失败题，再决定是否授权补跑失败评分。尤其keyword_002涉及条目拆分和参考答案范围，distractor_018涉及评分与人工阅读不一致。
4. 小规模实验优先只调整回答提示词：直接回答所问、避免主动列出无关技术、只引用支持对应结论的片段。选取各组代表题，固定检索配置比较；本次未执行。
5. 再单独研究Markdown列表／表格的片段边界，保留条目标签与解释，检查是否需要补邻接片段。索引重建会改变实验条件，应创建新实验记录并明确区别；本次未改动。
6. 分别试上下文数量（例如3与5）、hybrid权重，再考虑缩小rerank候选范围；每次仅改变一个因素。先确认实际候选数量与耗时组成，不把top_n直接当成模型输入候选数。
7. 选好质量与速度折中后再执行独立Load Test，测吞吐量、失败率和P99；当前评估不能支持并发性能承诺。

## 原始结果链接

- [vector_only](../data/experiments_history/20261004T013927_transformers_v4571_baseline_60_incremental_vecto_b2974b990a_output.yaml)
- [hybrid](../data/experiments_history/20261004T023806_transformers_v4571_baseline_60_incremental_hybri_3c4f333e71_output.yaml)
- [hybrid_rerank](../data/experiments_history/20261004T034946_transformers_v4571_baseline_60_incremental_hybri_cae3f19fb5_output.yaml)
