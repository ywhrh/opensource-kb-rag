# Golden dataset — English / Chinese reading edition

# 评估测试集——中英对照阅读版

60 questions: 20 semantic, 20 keyword, and 20 distractor. Questions and reference answers are translated below their English originals. API names, enum values and code retain their original spelling.

共 60 题：语义题、关键词题、干扰辨析题各 20 道。问题和参考答案的中文译文位于英文原文下方。API 名称、枚举值和代码保留原样。

This reading edition does not replace the English runtime JSON. It is outside the document corpus and must not be indexed. References remain a source-grounded draft pending independent human review; no model evaluation is implied.

这份阅读版不会替换评估用的英文 JSON。它位于文档库之外，不应建立检索索引。参考答案仍为依据文档编写、待独立人工审核的初稿，不代表已经进行模型评估。

## semantic_001

**Question**

Our deployment environment cannot reach the internet. How can we prepare a pretrained model beforehand and ensure loading does not contact the hosting service?

**问题（中文）**

我们的部署环境无法连接互联网。如何提前准备预训练模型，并确保加载时不访问托管服务？

**Reference answer**

Download the model repository and cache its files before going offline, for example with snapshot_download. Set HF_HUB_OFFLINE=1 to prevent HTTP calls to the Hub, or use from_pretrained(..., local_files_only=True) to load only locally cached files. Offline operation requires the needed files to already be available.

**参考答案（中文）**

离线前先下载模型仓库并缓存所需文件，例如使用 snapshot_download。设置 HF_HUB_OFFLINE=1 可阻止向 Hub 发起 HTTP 请求；也可以使用 from_pretrained(..., local_files_only=True)，只加载本地缓存文件。离线运行的前提是所需文件已经准备好。

## semantic_002

**Question**

A pretrained model cannot fit entirely on our graphics card. How can loading spread its weights across available hardware instead of requiring everything on that card?

**问题（中文）**

一个预训练模型无法完整放进我们的显卡。如何在加载时把权重分配到可用硬件上，而不是要求全部放在这张显卡上？

**Reference answer**

Load the model with from_pretrained(..., device_map="auto") to enable Accelerate Big Model Inference. Weights are dispatched to the fastest devices first, normally GPUs, and remaining weights can be offloaded to CPU and disk. This also avoids retaining separate random and pretrained copies during loading.

**参考答案（中文）**

使用 from_pretrained(..., device_map="auto") 加载模型，启用 Accelerate 的 Big Model Inference。权重优先分配到最快的设备，通常是 GPU；剩余权重可以卸载到 CPU 和磁盘。这种加载方式还可避免同时保留随机初始化权重和预训练权重两份副本。

## semantic_003

**Question**

Why can a text model handle an unfamiliar long word without keeping every possible word in its vocabulary?

**问题（中文）**

为什么文本模型不需要在词表里保存所有可能的单词，也能处理一个陌生的长单词？

**Reference answer**

Subword tokenization represents rare or unseen words as combinations of known subwords. This keeps vocabulary size manageable while retaining meaningful representations, rather than requiring every whole word to be present or treating text only as individual characters.

**参考答案（中文）**

子词分词将罕见词或未见过的词拆成已知子词的组合。这样既能控制词表规模，又能保留有意义的表示，而不必收录每个完整单词，也不必只按单个字符处理文本。

## semantic_004

**Question**

In a user-facing application, people currently see nothing until an entire answer finishes. What documented generation feature can make output appear progressively?

**问题（中文）**

在面向用户的应用中，目前必须等整段答案生成完才能看到内容。文档中的哪项生成功能可以让输出逐步显示？

**Reference answer**

Use streaming. Construct a TextStreamer with the tokenizer and pass it as the streamer argument to model.generate. It returns text as generation progresses rather than waiting for the complete response, reducing perceived latency.

**参考答案（中文）**

使用流式输出。用 tokenizer 创建 TextStreamer，并将其作为 streamer 参数传给 model.generate。它会随着生成过程返回文本，而不是等完整回答结束，从而降低用户感知到的延迟。

## semantic_005

**Question**

A conversational model sometimes continues the user’s sentence instead of replying. How can we make the formatted conversation indicate that the next text should be an assistant response?

**问题（中文）**

对话模型有时会续写用户的句子，而不是回答。如何让格式化后的对话表明，接下来应该生成助手的回复？

**Reference answer**

Use the model’s chat template with add_generation_prompt=True to append tokens marking the start of an assistant response. This tells the model to reply instead of continuing the user message. Some models do not need such a prefix, so the option has no effect for those templates.

**参考答案（中文）**

使用模型的聊天模板，并设置 add_generation_prompt=True，追加标记助手回复开始的 token。这能提示模型作答，而不是继续用户的消息。有些模型不需要这样的前缀，因此该选项对相应模板没有作用。

## semantic_006

**Question**

We want to adapt a large pretrained model on a consumer GPU and distribute only a small task-specific addition. What approach does the documentation recommend, and why does it save memory?

**问题（中文）**

我们想在消费级 GPU 上适配一个大型预训练模型，并且只分发一个很小的任务专用组件。文档推荐什么方法？它为什么能节省显存？

**Reference answer**

Use parameter-efficient fine-tuning with adapters, such as LoRA. Only a small number of extra parameters are trained on top of the pretrained base model. The GPU does not need optimizer states and gradients for the base model, and the lightweight adapters are convenient to store, share and load.

**参考答案（中文）**

使用带适配器的参数高效微调，例如 LoRA。只在预训练基础模型之上训练少量额外参数。GPU 无需保存基础模型的优化器状态和梯度；轻量的适配器也便于存储、分享和加载。

## semantic_007

**Question**

Our training examples fit only in small batches, but we want updates equivalent to using a larger batch. How can we achieve that without placing the whole larger batch in memory at once?

**问题（中文）**

我们的训练样本只能以小批次放入显存，但希望参数更新相当于使用更大的批次。如何在不同时放入整个大批次的情况下实现？

**Reference answer**

Use gradient accumulation: accumulate gradients across multiple mini-batches before updating model parameters. This produces a larger effective batch size while processing smaller batches. It can slow training because multiple forward and backward passes are needed for each update.

**参考答案（中文）**

使用梯度累积：先在多个小批次上累积梯度，再更新模型参数。这可以在处理小批次的同时获得更大的有效批次大小。每次更新需要多次前向和反向计算，因此训练可能变慢。

## semantic_008

**Question**

Intermediate activations are consuming too much memory during training. What technique can reduce their storage, and what cost should we expect?

**问题（中文）**

训练时，中间激活占用了太多显存。什么技术能减少它们的存储？需要付出什么代价？

**Reference answer**

Enable gradient checkpointing. Store only some intermediate activations and recompute the remaining ones when needed during the backward pass. This reduces activation memory at the cost of slower training; the guide describes an approximately 20 percent slowdown.

**参考答案（中文）**

启用梯度检查点。只保存部分中间激活，在反向传播需要时重新计算其余激活。这能减少激活占用的显存，但训练速度会降低；文档描述的减速幅度约为 20%。

## semantic_009

**Question**

An optimized model is slow on its first call but much faster on subsequent calls. What documented initialization cost can explain this pattern?

**问题（中文）**

一个经过优化的模型首次调用很慢，后续调用却快得多。文档中的哪种初始化开销能解释这种现象？

**Reference answer**

torch.compile needs to compile the model initially, so the initial call is slow. Later calls can reuse the compiled model rather than compiling again, making them faster. The one-time compilation cost should be distinguished from subsequent execution performance.

**参考答案（中文）**

torch.compile 最初需要编译模型，因此第一次调用较慢。后续调用可以复用已编译的模型，无需再次编译，所以更快。应将一次性的编译开销和后续执行性能区分开。

## semantic_010

**Question**

A model’s weight storage is too large for deployment. What general compression approach can lower its memory requirements, and what property is it trying to retain?

**问题（中文）**

模型权重的存储开销太大，不便部署。哪种通用压缩方法能降低内存需求？它试图保留什么性质？

**Reference answer**

Quantization stores model weights in lower precision to reduce memory requirements while trying to retain as much accuracy as possible. Depending on the method, weights may use lower-precision floating point or integer representations such as int8 or int4. Methods differ in calibration requirements and other trade-offs.

**参考答案（中文）**

量化以更低精度存储模型权重，在降低内存需求的同时尽量保留准确率。不同方法可能采用低精度浮点数，或 int8、int4 等整数表示。各方法的校准需求及其他取舍有所不同。

## semantic_011

**Question**

We want to restore a compressed model to its original precision. What operation is documented, and what must we consider before using it?

**问题（中文）**

我们希望将压缩后的模型恢复到原来的精度。文档介绍了什么操作？使用前需要考虑什么？

**Reference answer**

Call the model’s dequantize method to restore the original precision. Ensure enough GPU memory is available to hold the dequantized model, and note that dequantization may result in some quality loss.

**参考答案（中文）**

调用模型的 dequantize 方法恢复原来的精度。确保 GPU 有足够显存容纳反量化后的模型，并注意反量化可能带来一定的质量损失。

## semantic_012

**Question**

We need to deploy a model with a runtime that is separate from the original training framework. What export format and supporting library are described for this purpose?

**问题（中文）**

我们需要用独立于原训练框架的运行时部署模型。文档为此介绍了什么导出格式和支持库？

**Reference answer**

Export the model to ONNX, an open standard for representing models through common operators and a computational graph across frameworks. The Optimum library provides export support, and ONNX Runtime can run the exported model through task-specific ORTModel classes.

**参考答案（中文）**

将模型导出为 ONNX。它是一种开放标准，使用通用算子和计算图跨框架表示模型。Optimum 库提供导出支持；ONNX Runtime 可以通过对应任务的 ORTModel 类运行导出的模型。

## semantic_013

**Question**

Our preprocessing is a bottleneck because we encode text one sentence at a time. How should we organize it, and which tokenizer implementation is especially useful?

**问题（中文）**

我们逐句编码文本，导致预处理成为瓶颈。应该怎样组织预处理？哪种 tokenizer 实现尤其适合？

**Reference answer**

Preprocess batches of text rather than one sentence at a time. Fast tokenizers are Rust-based implementations and are especially effective at parallelizing batched tokenization; they also provide additional alignment methods compared with Python tokenizers.

**参考答案（中文）**

批量预处理文本，而不是每次只处理一句。Fast tokenizer 基于 Rust，尤其擅长并行处理批量分词；相比 Python tokenizer，它还提供额外的对齐方法。

## semantic_014

**Question**

A training job stopped unexpectedly, but its saved checkpoints are still available. How can we continue from saved progress rather than restarting from scratch?

**问题（中文）**

一个训练任务意外停止，但保存的检查点仍然存在。怎样从已保存的进度继续，而不是从头开始？

**Reference answer**

Call trainer.train with resume_from_checkpoint=True to resume from the last checkpoint, or pass a specific checkpoint path to resume from that checkpoint. Saved checkpoints are intended to support recovery and continuation of training.

**参考答案（中文）**

调用 trainer.train，并设置 resume_from_checkpoint=True，从最后一个检查点恢复；也可以传入具体检查点的路径，从指定位置恢复。保存检查点就是为了支持训练恢复和继续执行。

## semantic_015

**Question**

We loaded a general pretrained encoder for review classification and received a warning that the new classification layer has uninitialized weights. Why is this expected, and what should we do?

**问题（中文）**

我们加载了一个通用预训练编码器来做评论分类，随后看到新分类层的权重尚未初始化的警告。为什么这是预期行为？应该怎么办？

**Reference answer**

The pretrained head is replaced by a randomly initialized classification head for the new task. That head must be fine-tuned on the classification dataset before it can produce meaningful predictions. Load the sequence-classification model with the appropriate number of labels and train it.

**参考答案（中文）**

原预训练头被替换为面向新任务、随机初始化的分类头。必须在分类数据集上微调该分类头，才能产生有意义的预测。加载序列分类模型时设置正确的标签数量，并训练模型。

## semantic_016

**Question**

We found a published comparison in which one compression method is faster. How should we decide whether that method meets our own deployment requirements?

**问题（中文）**

我们看到一份公开对比，其中某种压缩方法速度更快。如何判断它是否满足我们自己的部署要求？

**Reference answer**

Benchmark the quantized model’s accuracy and speed on your specific task and hardware instead of treating the published comparison as a universal guarantee. Check whether the measured results meet your own requirements and consult the method’s usage documentation.

**参考答案（中文）**

在自己的任务和硬件上测试量化模型的准确率与速度，不要把公开对比当作普遍保证。检查实际测量结果是否符合自己的要求，并查阅该方法的使用文档。

## semantic_017

**Question**

Long conversations are exhausting GPU memory because their accumulated attention state is large. How can we keep less of that state on the GPU, and what performance trade-off follows?

**问题（中文）**

长对话积累的注意力状态很大，导致 GPU 显存耗尽。怎样减少 GPU 上保留的状态？会带来什么性能取舍？

**Reference answer**

Use cache offloading. Most layers’ KV caches are moved to CPU, while the current layer’s cache is maintained on GPU and the next layer is prefetched. This saves GPU memory, but transfers between CPU and GPU can reduce generation throughput.

**参考答案（中文）**

使用缓存卸载。大多数层的 KV 缓存被移到 CPU，GPU 上保留当前层的缓存，并预取下一层的缓存。这能节省显存，但 CPU 与 GPU 之间的数据传输可能降低生成吞吐量。

## semantic_018

**Question**

Our generation requests have very different lengths. Why might reserving one large fixed attention-state buffer be inefficient even though it helps compilation?

**问题（中文）**

我们的生成请求长度差异很大。为什么预留一个很大的固定注意力状态缓冲区，即使有助于编译，也可能效率不高？

**Reference answer**

A StaticCache preallocates a maximum size and supports compiling the decoding stage. However, unused positions are masked and still cause wasted attention computation, so a large fixed cache can be inefficient for many short requests mixed with a long request. It is a trade-off that is better suited to similar-length sequences.

**参考答案（中文）**

StaticCache 预分配最大容量，并支持编译解码阶段。但是未使用的位置虽然被掩码屏蔽，仍会造成注意力计算浪费。如果大量短请求与长请求混在一起，大型固定缓存可能效率不高。这种取舍更适合长度相近的序列。

## semantic_019

**Question**

We want to use an optimizer supplied by an external library with the standard training loop. What integration steps does the guide describe?

**问题（中文）**

我们想在标准训练循环中使用外部库提供的优化器。文档介绍了哪些接入步骤？

**Reference answer**

Install the library that supplies the optimizer and choose its identifier with the optim parameter in TrainingArguments. Depending on the optimizer, configure additional settings such as optim_target_modules or optim_args as documented, then use those training arguments with Trainer.

**参考答案（中文）**

安装提供优化器的库，并通过 TrainingArguments 的 optim 参数选择其标识符。根据具体优化器，按文档配置 optim_target_modules、optim_args 等额外选项，然后把这些训练参数交给 Trainer。

## semantic_020

**Question**

A custom attention implementation needs a different attention-mask format. What additional registration is needed so the model can construct the appropriate mask?

**问题（中文）**

自定义注意力实现需要不同格式的注意力掩码。还需要注册什么，才能让模型构造正确的掩码？

**Reference answer**

Register a corresponding mask function with AttentionMaskInterface in addition to the custom attention function. The mask interface adapts mask creation to the selected attention implementation. If no matching mask function is registered, mask creation is skipped.

**参考答案（中文）**

除了注册自定义注意力函数，还要通过 AttentionMaskInterface 注册对应的掩码函数。该接口根据所选注意力实现调整掩码创建方式。如果没有注册匹配的掩码函数，就会跳过掩码创建。

## keyword_001

**Question**

What do padding="longest" and padding="max_length" do when the tokenizer receives a single sequence?

**问题（中文）**

当 tokenizer 只接收一个序列时，padding="longest" 和 padding="max_length" 分别会做什么？

**Reference answer**

padding="longest" (or True) applies no padding to a single sequence. padding="max_length" still pads a single sequence to the specified max_length, or to the model’s maximum length if max_length is not provided.

**参考答案（中文）**

padding="longest"（或 True）不会对单个序列进行填充。padding="max_length" 仍会把单个序列填充到指定的 max_length；如果没有指定 max_length，则填充到模型的最大长度。

## keyword_002

**Question**

For a pair of sequences, which truncation value preserves the first sequence and truncates only the second?

**问题（中文）**

对于一对序列，哪个 truncation 值会保留第一个序列，只截断第二个？

**Reference answer**

Use truncation="only_second". It truncates only the second sequence of a pair to satisfy the specified max_length, or the model’s maximum accepted length when max_length is not provided.

**参考答案（中文）**

使用 truncation="only_second"。它只截断序列对中的第二个序列，使长度符合指定的 max_length；如果未指定，则使用模型允许的最大长度。

## keyword_003

**Question**

Which Pipeline parameter value returns only newly generated text instead of including the prompt?

**问题（中文）**

Pipeline 的哪个参数值能只返回新生成的文本，而不包含提示词？

**Reference answer**

Pass return_full_text=False to the text-generation pipeline. It returns the generated continuation rather than the prompt plus generated text.

**参考答案（中文）**

向文本生成 pipeline 传入 return_full_text=False。它会返回生成的续写部分，而不是提示词和生成文本的组合。

## keyword_004

**Question**

What return_timestamps value requests word-level timestamps from an automatic speech recognition pipeline?

**问题（中文）**

自动语音识别 pipeline 中，return_timestamps 的什么值会请求单词级时间戳？

**Reference answer**

Use return_timestamps="word". It requests timestamps indicating when each word was spoken.

**参考答案（中文）**

使用 return_timestamps="word"，请求每个单词被说出的时间戳。

## keyword_005

**Question**

What is the documented purpose of dtype="auto" when loading pretrained model weights?

**问题（中文）**

加载预训练权重时，文档中 dtype="auto" 的作用是什么？

**Reference answer**

dtype="auto" initializes weights directly in the data type in which they are stored. This can avoid loading them twice, first in PyTorch’s default float32 and then in the desired data type.

**参考答案（中文）**

dtype="auto" 直接按权重存储时的数据类型初始化权重。这可以避免先以 PyTorch 默认的 float32 加载，再以目标数据类型加载第二次。

## keyword_006

**Question**

What is the documented default max_shard_size in save_pretrained, and what does that parameter control?

**问题（中文）**

save_pretrained 中，文档记录的 max_shard_size 默认值是什么？这个参数控制什么？

**Reference answer**

max_shard_size defaults to 5GB for each checkpoint shard. It controls the shard size used when saving model weights, helping the resulting checkpoint fit memory-limited environments.

**参考答案（中文）**

max_shard_size 默认让每个检查点分片的大小为 5GB。它控制保存模型权重时的分片大小，帮助保存的检查点适用于内存受限的环境。

## keyword_007

**Question**

Which tokenizer return_tensors value requests PyTorch tensors?

**问题（中文）**

tokenizer 的 return_tensors 取什么值，会请求返回 PyTorch 张量？

**Reference answer**

Set return_tensors="pt" when calling the tokenizer to request PyTorch tensors.

**参考答案（中文）**

调用 tokenizer 时设置 return_tensors="pt"，即可请求 PyTorch 张量。

## keyword_008

**Question**

After apply_chat_template(tokenize=False), what add_special_tokens value should be used if the formatted string is tokenized later?

**问题（中文）**

调用 apply_chat_template(tokenize=False) 后，如果之后还要对格式化字符串分词，add_special_tokens 应设为什么？

**Reference answer**

Set add_special_tokens=False. The chat template should already contain the required special tokens, so adding them again can duplicate tokens and hurt model performance.

**参考答案（中文）**

设置 add_special_tokens=False。聊天模板应已包含所需特殊 token，再添加一次可能造成重复并影响模型性能。

## keyword_009

**Question**

What do raise_exception(msg) and strftime_now(format_str) do inside a chat template?

**问题（中文）**

聊天模板中的 raise_exception(msg) 和 strftime_now(format_str) 分别做什么？

**Reference answer**

raise_exception(msg) raises a TemplateException, useful for debugging or reporting incorrect template usage. strftime_now(format_str) retrieves the current date and time formatted with the supplied format string.

**参考答案（中文）**

raise_exception(msg) 抛出 TemplateException，可用于调试或报告模板使用错误。strftime_now(format_str) 按给定格式字符串返回当前日期和时间。

## keyword_010

**Question**

What does max_new_tokens control in text generation?

**问题（中文）**

文本生成中的 max_new_tokens 控制什么？

**Reference answer**

max_new_tokens controls the maximum number of newly generated tokens. Set it explicitly when a longer response is needed instead of relying on a potentially small generation default.

**参考答案（中文）**

max_new_tokens 控制新生成 token 的最大数量。需要更长回答时应显式设置，不要依赖可能较小的默认生成长度。

## keyword_011

**Question**

What settings enable multinomial sampling rather than beam search in generate?

**问题（中文）**

在 generate 中，哪些设置启用多项式采样，而不是束搜索？

**Reference answer**

Set do_sample=True and num_beams=1. This samples tokens from the model’s probability distribution without maintaining multiple beams.

**参考答案（中文）**

设置 do_sample=True 和 num_beams=1。这样会从模型的概率分布中采样 token，而不维护多个候选束。

## keyword_012

**Question**

Which cache_implementation values enable offloading for dynamic and static caches respectively?

**问题（中文）**

动态缓存和静态缓存分别用哪个 cache_implementation 值启用卸载？

**Reference answer**

Use cache_implementation="offloaded" for the dynamic cache and cache_implementation="offloaded_static" for the static cache. These can be configured in GenerationConfig or generate.

**参考答案（中文）**

动态缓存使用 cache_implementation="offloaded"；静态缓存使用 cache_implementation="offloaded_static"。可以在 GenerationConfig 或 generate 中配置。

## keyword_013

**Question**

What attention-mask shape is required when implementing an iterative forward loop with a KV cache?

**问题（中文）**

自己实现带 KV 缓存的迭代 forward 循环时，注意力掩码需要什么形状？

**Reference answer**

The attention mask must have shape (batch_size, past_kv_length + new_tokens_length), covering both previously cached tokens and current tokens.

**参考答案（中文）**

注意力掩码的形状必须是 (batch_size, past_kv_length + new_tokens_length)，同时覆盖之前缓存的 token 和当前 token。

## keyword_014

**Question**

Which TrainingArguments setting in the classification example reports evaluation accuracy at the end of each epoch?

**问题（中文）**

分类示例中的哪个 TrainingArguments 设置，会在每个 epoch 结束时报告评估准确率？

**Reference answer**

Use eval_strategy="epoch" in TrainingArguments. Supply a compute_metrics function to Trainer to compute the classification metric.

**参考答案（中文）**

在 TrainingArguments 中使用 eval_strategy="epoch"，并向 Trainer 提供 compute_metrics 函数来计算分类指标。

## keyword_015

**Question**

What does ZeRO stage 1 shard, and what extra states does ZeRO stage 2 shard?

**问题（中文）**

ZeRO stage 1 对什么进行分片？stage 2 额外对什么状态进行分片？

**Reference answer**

ZeRO-1 shards optimizer states across GPUs. ZeRO-2 shards both optimizer states and gradient states across GPUs.

**参考答案（中文）**

ZeRO-1 在 GPU 之间对优化器状态进行分片；ZeRO-2 同时对优化器状态和梯度状态进行分片。

## keyword_016

**Question**

How can set_attn_implementation switch an already loaded model back to SDPA?

**问题（中文）**

如何用 set_attn_implementation 将已加载的模型切换回 SDPA？

**Reference answer**

Call model.set_attn_implementation("sdpa"). This changes the attention function without needing to reload the model.

**参考答案（中文）**

调用 model.set_attn_implementation("sdpa")。它会切换注意力函数，无需重新加载模型。

## keyword_017

**Question**

Which BitsAndBytesConfig option enables nested four-bit quantization, and how much extra storage does the guide say it saves?

**问题（中文）**

BitsAndBytesConfig 的哪个选项启用嵌套四位量化？文档称它额外节省多少存储？

**Reference answer**

Set bnb_4bit_use_double_quant=True alongside load_in_4bit=True. Nested quantization quantizes already quantized weights again and saves an additional 0.4 bits per parameter.

**参考答案（中文）**

在 load_in_4bit=True 的同时设置 bnb_4bit_use_double_quant=True。嵌套量化会再次量化已经量化的权重，每个参数额外节省 0.4 bit。

## keyword_018

**Question**

What value of GPTQConfig.backend activates the Marlin kernel, and what bit width does that kernel support?

**问题（中文）**

GPTQConfig.backend 取什么值会启用 Marlin 内核？该内核支持多少位的权重？

**Reference answer**

Set backend="marlin" in GPTQConfig. The documented Marlin kernel supports four-bit GPTQ inference and is optimized for NVIDIA A100 GPUs.

**参考答案（中文）**

在 GPTQConfig 中设置 backend="marlin"。文档介绍的 Marlin 内核支持四位 GPTQ 推理，并针对 NVIDIA A100 GPU 进行了优化。

## keyword_019

**Question**

What do AwqConfig.do_fuse and fuse_max_seq_len configure?

**问题（中文）**

AwqConfig.do_fuse 和 fuse_max_seq_len 分别配置什么？

**Reference answer**

do_fuse=True enables fused AWQ modules. fuse_max_seq_len is the total sequence length, including both the input context and expected generated text; choose a sufficiently large value.

**参考答案（中文）**

do_fuse=True 启用融合的 AWQ 模块。fuse_max_seq_len 是总序列长度，包含输入上下文和预计生成的文本，因此应选择足够大的值。

## keyword_020

**Question**

What input-padding control is recommended with static KV cache and torch.compile to reduce shape-related recompilations?

**问题（中文）**

使用静态 KV 缓存和 torch.compile 时，文档推荐用哪个输入填充控制来减少由形状变化造成的重新编译？

**Reference answer**

Left-pad inputs using pad_to_multiple_of to a limited set of values. This reduces changes in input shapes that can trigger recompilation.

**参考答案（中文）**

使用 pad_to_multiple_of 对输入进行左侧填充，让长度落在有限的一组取值上。这样可以减少因输入形状变化而触发的重新编译。

## distractor_001

**Question**

A developer sets padding=True and claims this alone will shorten overlong inputs to the model limit. Is that correct, and what control is missing?

**问题（中文）**

开发者设置 padding=True，并声称仅靠它就能把过长输入缩短到模型限制以内。这正确吗？缺少什么控制？

**Reference answer**

No. Padding adds padding tokens to shorter sequences; it does not shorten long ones. Enable truncation=True, optionally with max_length, to truncate overlong inputs. Padding and truncation solve opposite length problems.

**参考答案（中文）**

不正确。Padding 给较短序列添加填充 token，不会缩短长序列。应启用 truncation=True，也可以指定 max_length，来截断过长输入。填充和截断处理的是方向相反的长度问题。

## distractor_002

**Question**

We want to continue an assistant message that already starts a JSON object. Should we enable add_generation_prompt and continue_final_message together to be safe?

**问题（中文）**

我们想继续生成一条已写出 JSON 对象开头的助手消息。为了稳妥，是否应该同时启用 add_generation_prompt 和 continue_final_message？

**Reference answer**

No. Use continue_final_message=True to continue the existing final message. add_generation_prompt starts a new assistant message, whereas continue_final_message removes end-of-sequence markers to continue one. Combining the two returns an error.

**参考答案（中文）**

不应该。使用 continue_final_message=True，继续现有的最后一条消息。add_generation_prompt 会开始一条新助手消息；continue_final_message 则移除序列结束标记，以续写现有消息。同时使用两者会报错。

## distractor_003

**Question**

For supervised training on completed conversations, should we keep add_generation_prompt=True because it is helpful when generating assistant replies?

**问题（中文）**

对已经完成的对话进行监督训练时，是否应该保留 add_generation_prompt=True，因为它在生成助手回复时很有用？

**Reference answer**

No. Apply the chat template while preprocessing training data, but set add_generation_prompt=False. The extra tokens that prompt a new assistant response are not helpful during training on the supplied conversations.

**参考答案（中文）**

不应该。预处理训练数据时应用聊天模板，但设置 add_generation_prompt=False。在已有对话上训练时，提示模型开始新助手回复的额外 token 并无帮助。

## distractor_004

**Question**

Is beam search the same as greedily choosing the highest-probability next token at every step? Explain the difference.

**问题（中文）**

束搜索是否等同于每一步都贪心地选择概率最高的下一个 token？请解释区别。

**Reference answer**

No. Greedy search selects the most likely next token at each step. Beam search tracks several candidate sequences and selects one with the highest overall probability, so it can retain a sequence whose early tokens are not locally the most likely.

**参考答案（中文）**

不等同。贪心搜索在每一步选择最可能的下一个 token；束搜索维护多个候选序列，并选择整体概率最高的序列，因此可以保留早期 token 并非局部最优的序列。

## distractor_005

**Question**

Can WatermarkDetector use an arbitrary WatermarkingConfig to detect text that was generated with another configuration?

**问题（中文）**

WatermarkDetector 能否使用任意 WatermarkingConfig，检测使用另一配置生成的文本？

**Reference answer**

No. WatermarkDetector must use the same WatermarkingConfig as the one used during generation. It detects the proportion of green tokens under that watermarking scheme.

**参考答案（中文）**

不能。WatermarkDetector 必须使用与生成时相同的 WatermarkingConfig。它会检测该水印方案下绿色 token 的比例。

## distractor_006

**Question**

Does constructing a model from its configuration give us the pretrained checkpoint’s learned weights?

**问题（中文）**

仅根据配置构造模型，会获得预训练检查点中已学习的权重吗？

**Reference answer**

No. A configuration describes the architecture and its attributes; constructing the model this way yields random weights that need training. Load learned checkpoint weights with from_pretrained instead.

**参考答案（中文）**

不会。配置描述架构及其属性；这样构造出来的模型具有随机权重，需要训练。应使用 from_pretrained 加载已经学习好的检查点权重。

## distractor_007

**Question**

Can we freely replace a pretrained model’s tokenizer with another tokenizer just because both produce input IDs?

**问题（中文）**

只要两个 tokenizer 都输出 input IDs，就能随意替换预训练模型原来的 tokenizer 吗？

**Reference answer**

No. The tokenizer’s vocabulary must match the vocabulary expected by the pretrained model. A custom tokenizer with a different vocabulary cannot be assumed interchangeable merely because it produces numeric input IDs.

**参考答案（中文）**

不能。Tokenizer 的词表必须匹配预训练模型所期望的词表。不能仅因为自定义 tokenizer 输出数值 ID，就认为不同词表的 tokenizer 可以互换。

## distractor_008

**Question**

We have two attached adapters. Does enable_adapters select only the second adapter in the same way as set_adapter("lora_2")?

**问题（中文）**

模型上有两个适配器。enable_adapters 是否和 set_adapter("lora_2") 一样，只选择第二个适配器？

**Reference answer**

No. set_adapter("lora_2") selects that adapter and disables the others. enable_adapters is a broader operation that enables all adapters attached to the model.

**参考答案（中文）**

不是。set_adapter("lora_2") 选择该适配器并禁用其他适配器；enable_adapters 是更广泛的操作，会启用模型上所有已附加的适配器。

## distractor_009

**Question**

Because cached attention state speeds up generation, should we also enable it during model training?

**问题（中文）**

缓存注意力状态可以加速生成，所以是否也应该在模型训练时启用它？

**Reference answer**

No. The documentation says caching should be used only for inference and may cause unexpected errors if enabled during training.

**参考答案（中文）**

不应该。文档明确说明缓存只能用于推理，在训练时启用可能导致意外错误。

## distractor_010

**Question**

A reused KV cache contains ten tokens, including padding positions. Should cache_position for the next token subtract the padding count?

**问题（中文）**

复用的 KV 缓存包含十个 token，其中包括填充位置。下一个 token 的 cache_position 是否应该减去填充数量？

**Reference answer**

No. cache_position is unaffected by padding and advances once per token. For a cache containing ten tokens, the next position is torch.tensor([10]), regardless of padding tokens.

**参考答案（中文）**

不应该。cache_position 不受填充影响，每个 token 都会让它增加一个位置。缓存包含十个 token 时，下一个位置是 torch.tensor([10])，与其中是否有填充 token 无关。

## distractor_011

**Question**

Is the dynamically growing cache the preferred choice when we specifically need a fixed shape for compiling decoding?

**问题（中文）**

如果我们明确需要固定形状来编译解码过程，动态增长的缓存是不是首选？

**Reference answer**

No. DynamicCache grows as generation progresses. StaticCache preallocates a maximum capacity and supports compiling decoding, at the cost of unused masked positions and potentially wasted computation.

**参考答案（中文）**

不是。DynamicCache 随生成过程增长；StaticCache 预分配最大容量并支持编译解码，但未使用的掩码位置可能造成计算浪费。

## distractor_012

**Question**

If we enable a static KV cache and compile the model, can we freely change batch size or increase maximum output length without triggering recompilation?

**问题（中文）**

启用静态 KV 缓存并编译模型后，能否任意改变批次大小或增大最大输出长度，而不触发重新编译？

**Reference answer**

No. Changing batch size or increasing maximum output length between calls can reinitialize and recompile the cache. generate tries to reuse the cache to avoid recompilation, so workload shape changes still matter.

**参考答案（中文）**

不能。调用之间改变批次大小或增大最大输出长度，可能让缓存重新初始化并重新编译。generate 会尝试复用缓存来避免重新编译，因此工作负载的形状变化仍然重要。

## distractor_013

**Question**

We chose SDPA but also request output_attentions=True. Should we assume the same fast attention implementation is still being used?

**问题（中文）**

我们选择了 SDPA，同时请求 output_attentions=True。是否可以认为仍在使用同一个快速注意力实现？

**Reference answer**

No. The guide lists output_attentions=True and head_mask as unsupported by SDPA and says Transformers warns and falls back to the slower eager implementation for such options.

**参考答案（中文）**

不可以。文档将 output_attentions=True 和 head_mask 列为 SDPA 不支持的选项，并说明 Transformers 会发出警告，退回较慢的 eager 实现。

## distractor_014

**Question**

Can we combine fused AWQ modules with FlashAttention2 to stack both optimizations?

**问题（中文）**

能否把融合的 AWQ 模块和 FlashAttention2 组合起来，叠加两种优化？

**Reference answer**

No. The AWQ documentation explicitly says fused modules cannot be combined with optimization techniques such as FlashAttention2. Choose a compatible configuration instead of enabling both together.

**参考答案（中文）**

不能。AWQ 文档明确说明融合模块不能与 FlashAttention2 等优化技术组合。应选择兼容配置，而不是同时启用两者。

## distractor_015

**Question**

Can the Marlin backend perform the initial GPTQ quantization of a full-precision model, as well as accelerate its later inference?

**问题（中文）**

Marlin 后端能否既负责将全精度模型初次量化为 GPTQ，又加速后续推理？

**Reference answer**

No. The documented Marlin backend is for quantized inference only and does not support model quantization. Initial quantization must be performed by a supported quantization workflow before Marlin is used for inference.

**参考答案（中文）**

不能。文档中的 Marlin 后端仅用于量化模型的推理，不支持模型量化。应先通过支持的量化流程完成初次量化，再使用 Marlin 推理。

## distractor_016

**Question**

After enabling eight-bit CPU offloading, are the offloaded CPU weights also stored as eight-bit integers?

**问题（中文）**

启用八位 CPU 卸载后，被卸载到 CPU 的权重也以八位整数存储吗？

**Reference answer**

No. The bitsandbytes offloading section says weights dispatched to CPU are stored in float32 and are not converted to eight-bit.

**参考答案（中文）**

不是。Bitsandbytes 的卸载章节说明，分配到 CPU 的权重以 float32 存储，不会转换成八位。

## distractor_017

**Question**

Do bnb_4bit_quant_type="nf4" and bnb_4bit_compute_dtype=torch.bfloat16 control the same thing?

**问题（中文）**

bnb_4bit_quant_type="nf4" 和 bnb_4bit_compute_dtype=torch.bfloat16 控制的是同一件事吗？

**Reference answer**

No. bnb_4bit_quant_type="nf4" selects the four-bit weight quantization representation, which the guide recommends for normally distributed weights when training four-bit base models. bnb_4bit_compute_dtype selects the computation data type, such as bfloat16; it does not select the four-bit storage format.

**参考答案（中文）**

不是。bnb_4bit_quant_type="nf4" 选择四位权重的量化表示，文档推荐在训练具有正态分布权重的四位基础模型时使用它。bnb_4bit_compute_dtype 选择计算时的数据类型，例如 bfloat16，而不是四位存储格式。

## distractor_018

**Question**

Are GPTQ and AWQ on-the-fly methods that require no calibration dataset, just like the no-calibration approaches listed in the guide?

**问题（中文）**

GPTQ 和 AWQ 是否和文档列出的无需校准方法一样，能即时量化而不需要校准数据集？

**Reference answer**

No. The selecting guide groups GPTQ/GPTQModel and AWQ under Calibration-based Quantization, which requires an upfront calibration step using a dataset. bitsandbytes, HQQ and torchao are listed under No Calibration Required.

**参考答案（中文）**

不是。选择指南把 GPTQ/GPTQModel 和 AWQ 归为基于校准的量化，需要先用数据集进行校准。Bitsandbytes、HQQ 和 torchao 则列在无需校准的方法中。

## distractor_019

**Question**

We serve individual low-latency requests on a CPU. Should we increase Pipeline batch_size because batching is guaranteed to make inference faster?

**问题（中文）**

我们在 CPU 上处理单个低延迟请求。因为批处理一定让推理更快，是否应该增大 Pipeline 的 batch_size？

**Reference answer**

No. Pipeline batch inference may help, especially on a GPU, but faster performance is not guaranteed. The guide advises against batching for latency-constrained live products and CPU use, and says to measure on the actual model, data and hardware.

**参考答案（中文）**

不应该这样假定。Pipeline 批处理可能有帮助，尤其是在 GPU 上，但不保证更快。文档不建议对延迟敏感的实时产品和 CPU 场景使用批处理，并要求在实际模型、数据和硬件上测量。

## distractor_020

**Question**

With fullgraph=True, will torch.compile quietly split the model into several graphs when it encounters a graph break?

**问题（中文）**

设置 fullgraph=True 后，torch.compile 遇到计算图中断时，会悄悄把模型拆成多个图吗？

**Reference answer**

No. fullgraph=True attempts to compile the entire model into one graph. torch.compile raises an error if it encounters a graph break instead of quietly accepting several graphs.

**参考答案（中文）**

不会。fullgraph=True 要求尝试把整个模型编译成单个图。遇到计算图中断时，torch.compile 会报错，而不是默默接受多个图。

