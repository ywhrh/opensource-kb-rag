# Golden dataset — English / Chinese reading edition

# 评估测试集——中英对照阅读版

60 questions: 20 semantic, 20 keyword, and 20 distractor. Questions and reference answers are translated below their English originals. API names, enum values and code retain their original spelling.

共 60 题：语义题、关键词题、干扰辨析题各 20 道。问题和参考答案的中文译文位于英文原文下方。API 名称、枚举值和代码保留原样。

This reading edition does not replace the English runtime JSON. It is outside the document corpus and must not be indexed. References remain a source-grounded draft pending independent human review; no model evaluation is implied.

这份阅读版不会替换评估用的英文 JSON。它位于文档库之外，不应建立检索索引。参考答案仍为依据文档编写、待独立人工审核的初稿，不代表已经进行模型评估。

## semantic_001

**Focus**

Map an offline deployment requirement to preparation and loading controls.

**考查点（中文）**

将离线部署需求对应到文件准备和加载控制。

**Question**

Our deployment environment cannot reach the internet. How can we prepare a pretrained model beforehand and ensure loading does not contact the hosting service?

**问题（中文）**

我们的部署环境无法连接互联网。如何提前准备预训练模型，并确保加载时不访问托管服务？

**Reference answer**

Download the model repository and cache its files before going offline, for example with snapshot_download. Set HF_HUB_OFFLINE=1 to prevent HTTP calls to the Hub, or use from_pretrained(..., local_files_only=True) to load only locally cached files. Offline operation requires the needed files to already be available.

**参考答案（中文）**

离线前先下载模型仓库并缓存所需文件，例如使用 snapshot_download。设置 HF_HUB_OFFLINE=1 可阻止向 Hub 发起 HTTP 请求；也可以使用 from_pretrained(..., local_files_only=True)，只加载本地缓存文件。离线运行的前提是所需文件已经准备好。

**Source:** [installation.md, lines 142–142](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/installation.md#L142)

**来源：** 同上文档第 142–142 行；以下译文对应所列原文。

**Original evidence**

````text
To use Transformers in an offline or firewalled environment requires the downloaded and cached files ahead of time. Download a model repository from the Hub with the [`~huggingface_hub.snapshot_download`] method.
````

**原文依据（中文译文）**

在离线或有防火墙限制的环境中使用 Transformers，需要提前下载并缓存文件。可以通过 huggingface_hub.snapshot_download 方法从 Hub 下载模型仓库。

**Source:** [installation.md, lines 153–153](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/installation.md#L153)

**来源：** 同上文档第 153–153 行；以下译文对应所列原文。

**Original evidence**

````text
Set the environment variable `HF_HUB_OFFLINE=1` to prevent HTTP calls to the Hub when loading a model.
````

**原文依据（中文译文）**

设置环境变量 HF_HUB_OFFLINE=1，可在加载模型时阻止向 Hub 发起 HTTP 请求。

**Source:** [installation.md, lines 160–160](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/installation.md#L160)

**来源：** 同上文档第 160–160 行；以下译文对应所列原文。

**Original evidence**

````text
Another option for only loading cached files is to set `local_files_only=True` in [`~PreTrainedModel.from_pretrained`].
````

**原文依据（中文译文）**

只加载缓存文件的另一种方法，是在 PreTrainedModel.from_pretrained 中设置 local_files_only=True。

## semantic_002

**Focus**

Understand automatic placement from a memory-limited deployment scenario.

**考查点（中文）**

根据部署时的显存限制，理解自动设备分配。

**Question**

A pretrained model cannot fit entirely on our graphics card. How can loading spread its weights across available hardware instead of requiring everything on that card?

**问题（中文）**

一个预训练模型无法完整放进我们的显卡。如何在加载时把权重分配到可用硬件上，而不是要求全部放在这张显卡上？

**Reference answer**

Load the model with from_pretrained(..., device_map="auto") to enable Accelerate Big Model Inference. Weights are dispatched to the fastest devices first, normally GPUs, and remaining weights can be offloaded to CPU and disk. This also avoids retaining separate random and pretrained copies during loading.

**参考答案（中文）**

使用 from_pretrained(..., device_map="auto") 加载模型，启用 Accelerate 的 Big Model Inference。权重优先分配到最快的设备，通常是 GPU；剩余权重可以卸载到 CPU 和磁盘。这种加载方式还可避免同时保留随机初始化权重和预训练权重两份副本。

**Source:** [models.md, lines 200–200](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L200)

**来源：** 同上文档第 200–200 行；以下译文对应所列原文。

**Original evidence**

````text
Big Model Inference's second feature relates to how weights are loaded and dispatched in the model skeleton. Model weights are dispatched across all available devices, starting with the fastest device (usually the GPU) and then offloading any remaining weights to slower devices (CPU and hard drive).
````

**原文依据（中文译文）**

Big Model Inference 的第二项功能涉及权重如何加载和分配到模型骨架中。权重会分配到所有可用设备，优先使用最快的设备（通常是 GPU），然后将剩余权重卸载到较慢的设备（CPU 和硬盘）。

**Source:** [models.md, lines 204–204](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L204)

**来源：** 同上文档第 204–204 行；以下译文对应所列原文。

**Original evidence**

````text
Set [device_map](https://github.com/huggingface/transformers/blob/026a173a64372e9602a16523b8fae9de4b0ff428/src/transformers/modeling_utils.py#L3061) to `"auto"` to enable Big Model Inference.
````

**原文依据（中文译文）**

将 device_map 设置为 "auto"，启用 Big Model Inference。

**Source:** [models.md, lines 195–195](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L195)

**来源：** 同上文档第 195–195 行；以下译文对应所列原文。

**Original evidence**

````text
Randomly initialized weights are only created when the pretrained weights are loaded to avoid maintaining two copies of the model in memory at the same time. The maximum memory usage is only the size of the model.
````

**原文依据（中文译文）**

只有在加载预训练权重时才创建随机初始化权重，避免同时在内存中维护两份模型副本。最大内存占用仅为模型本身的大小。

## semantic_003

**Focus**

Explain the purpose of subword representation from an unseen-word scenario.

**考查点（中文）**

通过未见过的单词，理解子词表示的作用。

**Question**

Why can a text model handle an unfamiliar long word without keeping every possible word in its vocabulary?

**问题（中文）**

为什么文本模型不需要在词表里保存所有可能的单词，也能处理一个陌生的长单词？

**Reference answer**

Subword tokenization represents rare or unseen words as combinations of known subwords. This keeps vocabulary size manageable while retaining meaningful representations, rather than requiring every whole word to be present or treating text only as individual characters.

**参考答案（中文）**

子词分词将罕见词或未见过的词拆成已知子词的组合。这样既能控制词表规模，又能保留有意义的表示，而不必收录每个完整单词，也不必只按单个字符处理文本。

**Source:** [tokenizer_summary.md, lines 104–107](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/tokenizer_summary.md#L104)

**来源：** 同上文档第 104–107 行；以下译文对应所列原文。

**Original evidence**

````text
Subword tokenization allows the model to have a reasonable vocabulary size while being able to learn meaningful
context-independent representations. In addition, subword tokenization enables the model to process words it has never
seen before, by decomposing them into known subwords. For instance, the [`~transformers.BertTokenizer`] tokenizes
`"I have a new GPU!"` as follows:
````

**原文依据（中文译文）**

子词分词使模型在保持合理词表规模的同时，学习有意义且不依赖具体上下文的表示。它还能将从未见过的词拆成已知子词，使模型可以处理这些词。例如，BertTokenizer 对 "I have a new GPU!"（我有一块新 GPU！）的分词如下。

## semantic_004

**Focus**

Connect user-visible waiting to progressive output.

**考查点（中文）**

将用户等待问题对应到逐步输出。

**Question**

In a user-facing application, people currently see nothing until an entire answer finishes. What documented generation feature can make output appear progressively?

**问题（中文）**

在面向用户的应用中，目前必须等整段答案生成完才能看到内容。文档中的哪项生成功能可以让输出逐步显示？

**Reference answer**

Use streaming. Construct a TextStreamer with the tokenizer and pass it as the streamer argument to model.generate. It returns text as generation progresses rather than waiting for the complete response, reducing perceived latency.

**参考答案（中文）**

使用流式输出。用 tokenizer 创建 TextStreamer，并将其作为 streamer 参数传给 model.generate。它会随着生成过程返回文本，而不是等完整回答结束，从而降低用户感知到的延迟。

**Source:** [generation_features.md, lines 25–25](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L25)

**来源：** 同上文档第 25–25 行；以下译文对应所列原文。

**Original evidence**

````text
Streaming starts returning text as soon as it is generated so you don't have to wait to see the entire generated response all at once. It is important in user-facing applications because it reduces perceived latency and allows users to see the generation progression.
````

**原文依据（中文译文）**

流式输出会在文本一经生成时就开始返回，而不必等整段回答全部完成。它对面向用户的应用很重要，因为它能降低感知延迟，让用户看到生成进度。

**Source:** [generation_features.md, lines 34–34](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L34)

**来源：** 同上文档第 34–34 行；以下译文对应所列原文。

**Original evidence**

````text
Create an instance of [`TextStreamer`] with the tokenizer. Pass [`TextStreamer`] to the `streamer` parameter in [`~GenerationMixin.generate`] to stream the output one word at a time.
````

**原文依据（中文译文）**

用 tokenizer 创建 TextStreamer 实例，将其传入 GenerationMixin.generate 的 streamer 参数，就可以逐词流式输出。

## semantic_005

**Focus**

Diagnose a conversation boundary problem without naming its control in the question.

**考查点（中文）**

在问题中不直接点名参数，诊断对话边界问题。

**Question**

A conversational model sometimes continues the user’s sentence instead of replying. How can we make the formatted conversation indicate that the next text should be an assistant response?

**问题（中文）**

对话模型有时会续写用户的句子，而不是回答。如何让格式化后的对话表明，接下来应该生成助手的回复？

**Reference answer**

Use the model’s chat template with add_generation_prompt=True to append tokens marking the start of an assistant response. This tells the model to reply instead of continuing the user message. Some models do not need such a prefix, so the option has no effect for those templates.

**参考答案（中文）**

使用模型的聊天模板，并设置 add_generation_prompt=True，追加标记助手回复开始的 token。这能提示模型作答，而不是继续用户的消息。有些模型不需要这样的前缀，因此该选项对相应模板没有作用。

**Source:** [chat_templating.md, lines 131–133](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L131)

**来源：** 同上文档第 131–133 行；以下译文对应所列原文。

**Original evidence**

````text
You may have noticed the [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) argument in the above examples.
This argument adds tokens to the end of the chat that indicate the start of an `assistant` response. Remember: Beneath all the chat abstractions, chat models are still just language models that continue a sequence of tokens!
If you include tokens that tell it that it's now in an `assistant` response, it will correctly write a response, but if you don't include these tokens, the model may get confused and do something strange, like **continuing** the user's message instead of replying to it!
````

**原文依据（中文译文）**

你可能注意到了前面示例中的 add_generation_prompt 参数。它在对话末尾添加标记 assistant 回复开始的 token。请记住：在所有聊天抽象之下，对话模型仍然只是继续生成 token 序列的语言模型。如果添加了提示当前已进入 assistant 回复的 token，模型就能正确回复；否则它可能混淆，出现继续用户消息而不作答等行为。

**Source:** [chat_templating.md, lines 171–171](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L171)

**来源：** 同上文档第 171–171 行；以下译文对应所列原文。

**Original evidence**

````text
Not all models require generation prompts, and some models, like [Llama](./model_doc/llama), don't have any special tokens before the `assistant` response. In these cases, [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) has no effect.
````

**原文依据（中文译文）**

并非所有模型都需要生成提示。某些模型（例如 Llama）在 assistant 回复之前没有特殊 token，因此 add_generation_prompt 对它们没有作用。

## semantic_006

**Focus**

Relate limited training resources and small deliverables to adapters.

**考查点（中文）**

将有限训练资源和小型分发组件对应到适配器。

**Question**

We want to adapt a large pretrained model on a consumer GPU and distribute only a small task-specific addition. What approach does the documentation recommend, and why does it save memory?

**问题（中文）**

我们想在消费级 GPU 上适配一个大型预训练模型，并且只分发一个很小的任务专用组件。文档推荐什么方法？它为什么能节省显存？

**Reference answer**

Use parameter-efficient fine-tuning with adapters, such as LoRA. Only a small number of extra parameters are trained on top of the pretrained base model. The GPU does not need optimizer states and gradients for the base model, and the lightweight adapters are convenient to store, share and load.

**参考答案（中文）**

使用带适配器的参数高效微调，例如 LoRA。只在预训练基础模型之上训练少量额外参数。GPU 无需保存基础模型的优化器状态和梯度；轻量的适配器也便于存储、分享和加载。

**Source:** [peft.md, lines 16–16](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/peft.md#L16)

**来源：** 同上文档第 16–16 行；以下译文对应所列原文。

**Original evidence**

````text
[PEFT](https://huggingface.co/docs/peft/index), a library of parameter-efficient fine-tuning methods, enables training and storing large models on consumer GPUs. These methods only fine-tune a small number of extra model parameters, also known as adapters, on top of the pretrained model. A significant amount of memory is saved because the GPU doesn't need to store the optimizer states and gradients for the pretrained base model. Adapters are very lightweight, making it convenient to share, store, and load them.
````

**原文依据（中文译文）**

PEFT 是一个提供参数高效微调方法的库，使消费级 GPU 也能训练和存储大型模型。这些方法只在预训练模型之上微调少量额外参数，也称为适配器。GPU 无需保存基础模型的优化器状态和梯度，因此能显著节省显存。适配器很轻量，便于分享、存储和加载。

## semantic_007

**Focus**

Distinguish an effective training batch from a simultaneously resident batch.

**考查点（中文）**

区分有效训练批次和同时驻留显存的批次。

**Question**

Our training examples fit only in small batches, but we want updates equivalent to using a larger batch. How can we achieve that without placing the whole larger batch in memory at once?

**问题（中文）**

我们的训练样本只能以小批次放入显存，但希望参数更新相当于使用更大的批次。如何在不同时放入整个大批次的情况下实现？

**Reference answer**

Use gradient accumulation: accumulate gradients across multiple mini-batches before updating model parameters. This produces a larger effective batch size while processing smaller batches. It can slow training because multiple forward and backward passes are needed for each update.

**参考答案（中文）**

使用梯度累积：先在多个小批次上累积梯度，再更新模型参数。这可以在处理小批次的同时获得更大的有效批次大小。每次更新需要多次前向和反向计算，因此训练可能变慢。

**Source:** [perf_train_gpu_one.md, lines 63–63](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_train_gpu_one.md#L63)

**来源：** 同上文档第 63–63 行；以下译文对应所列原文。

**Original evidence**

````text
Gradient accumulation overcomes memory constraints - useful for fitting a very large model that otherwise wouldn't fit on a single GPU - by accumulating gradients over multiple mini-batches before updating the parameters. This reduces memory by storing fewer gradients and enables training with a larger *effective batch size* because usually, the parameters are updated from a single batch of data. Training can slow down though due to the additional forward and backward passes introduced by gradient accumulation.
````

**原文依据（中文译文）**

梯度累积通过在更新参数之前累积多个小批次的梯度，克服内存限制，适用于原本无法放入单个 GPU 的大模型。它通过保存更少的梯度来降低内存占用，并实现更大的有效批次大小；通常参数是根据单个批次更新的。但额外的前向和反向计算可能降低训练速度。

## semantic_008

**Focus**

Identify recomputation as the trade-off for activation storage.

**考查点（中文）**

理解用重新计算换取更低激活存储开销。

**Question**

Intermediate activations are consuming too much memory during training. What technique can reduce their storage, and what cost should we expect?

**问题（中文）**

训练时，中间激活占用了太多显存。什么技术能减少它们的存储？需要付出什么代价？

**Reference answer**

Enable gradient checkpointing. Store only some intermediate activations and recompute the remaining ones when needed during the backward pass. This reduces activation memory at the cost of slower training; the guide describes an approximately 20 percent slowdown.

**参考答案（中文）**

启用梯度检查点。只保存部分中间激活，在反向传播需要时重新计算其余激活。这能减少激活占用的显存，但训练速度会降低；文档描述的减速幅度约为 20%。

**Source:** [perf_train_gpu_one.md, lines 86–86](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_train_gpu_one.md#L86)

**来源：** 同上文档第 86–86 行；以下译文对应所列原文。

**Original evidence**

````text
Gradient checkpointing reduces memory usage by only storing some of the intermediate activations during the backward pass and recomputing the remaining activations. This avoids storing *all* of the intermediate activations from the forward pass, which can require a lot of memory overhead. However, it comes at the cost of slower training speed (~20%).
````

**原文依据（中文译文）**

梯度检查点只保存部分中间激活，并在反向传播中重新计算其余激活，以减少内存占用。这样无需保存前向传播中的全部中间激活，但训练速度会降低约 20%。

## semantic_009

**Focus**

Interpret cold-start latency rather than confusing it with steady-state inference.

**考查点（中文）**

解释冷启动延迟，避免与稳定运行时的推理性能混淆。

**Question**

An optimized model is slow on its first call but much faster on subsequent calls. What documented initialization cost can explain this pattern?

**问题（中文）**

一个经过优化的模型首次调用很慢，后续调用却快得多。文档中的哪种初始化开销能解释这种现象？

**Reference answer**

torch.compile needs to compile the model initially, so the initial call is slow. Later calls can reuse the compiled model rather than compiling again, making them faster. The one-time compilation cost should be distinguished from subsequent execution performance.

**参考答案（中文）**

torch.compile 最初需要编译模型，因此第一次调用较慢。后续调用可以复用已编译的模型，无需再次编译，所以更快。应将一次性的编译开销和后续执行性能区分开。

**Source:** [perf_torch_compile.md, lines 29–30](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_torch_compile.md#L29)

**来源：** 同上文档第 29–30 行；以下译文对应所列原文。

**Original evidence**

````text
> [!TIP]
> The initial call to torch.compile is slow because the model needs to be compiled. Subsequent calls to the compiled model are much faster because it doesn't need to compile again.
````

**原文依据（中文译文）**

提示：torch.compile 的首次调用较慢，因为需要编译模型。后续调用已编译模型时无需再次编译，因此会快得多。

## semantic_010

**Focus**

Explain weight precision reduction without requiring an algorithm name in the question.

**考查点（中文）**

在问题中不点名算法，解释降低权重精度的作用。

**Question**

A model’s weight storage is too large for deployment. What general compression approach can lower its memory requirements, and what property is it trying to retain?

**问题（中文）**

模型权重的存储开销太大，不便部署。哪种通用压缩方法能降低内存需求？它试图保留什么性质？

**Reference answer**

Quantization stores model weights in lower precision to reduce memory requirements while trying to retain as much accuracy as possible. Depending on the method, weights may use lower-precision floating point or integer representations such as int8 or int4. Methods differ in calibration requirements and other trade-offs.

**参考答案（中文）**

量化以更低精度存储模型权重，在降低内存需求的同时尽量保留准确率。不同方法可能采用低精度浮点数，或 int8、int4 等整数表示。各方法的校准需求及其他取舍有所不同。

**Source:** [quantization/overview.md, lines 19–19](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/overview.md#L19)

**来源：** 同上文档第 19–19 行；以下译文对应所列原文。

**Original evidence**

````text
Quantization lowers the memory requirements of loading and using a model by storing the weights in a lower precision while trying to preserve as much accuracy as possible. Weights are typically stored in full-precision (fp32) floating point representations, but half-precision (fp16 or bf16) are increasingly popular data types given the large size of models today. Some quantization methods can reduce the precision even further to integer representations, like int8 or int4.
````

**原文依据（中文译文）**

量化以更低精度存储权重，减少加载和使用模型的内存需求，同时尽可能保留准确率。权重通常采用全精度 fp32 浮点表示；随着模型规模增大，fp16 或 bf16 半精度也越来越常见。某些量化方法可进一步降低精度，使用 int8、int4 等整数表示。

**Source:** [quantization/overview.md, lines 21–21](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/overview.md#L21)

**来源：** 同上文档第 21–21 行；以下译文对应所列原文。

**Original evidence**

````text
Transformers supports many quantization methods, each with their pros and cons, so you can pick the best one for your specific use case. Some methods require calibration for greater accuracy and extreme compression (1-2 bits), while other methods work out of the box with on-the-fly quantization.
````

**原文依据（中文译文）**

Transformers 支持多种量化方法，各有优缺点，可根据具体场景选择。有些方法需要校准，以提高准确率并实现 1～2 bit 等极限压缩；另一些方法无需额外准备，可即时量化。

## semantic_011

**Focus**

Map restoration of compressed weights to the operation and its caveats.

**考查点（中文）**

将压缩权重的恢复需求对应到操作及其注意事项。

**Question**

We want to restore a compressed model to its original precision. What operation is documented, and what must we consider before using it?

**问题（中文）**

我们希望将压缩后的模型恢复到原来的精度。文档介绍了什么操作？使用前需要考虑什么？

**Reference answer**

Call the model’s dequantize method to restore the original precision. Ensure enough GPU memory is available to hold the dequantized model, and note that dequantization may result in some quality loss.

**参考答案（中文）**

调用模型的 dequantize 方法恢复原来的精度。确保 GPU 有足够显存容纳反量化后的模型，并注意反量化可能带来一定的质量损失。

**Source:** [quantization/bitsandbytes.md, lines 329–329](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L329)

**来源：** 同上文档第 329–329 行；以下译文对应所列原文。

**Original evidence**

````text
Once quantized, you can [`~PreTrainedModel.dequantize`] a model to the original precision but this may result in some quality loss. Make sure you have enough GPU memory to fit the dequantized model.
````

**原文依据（中文译文）**

量化后的模型可以通过 PreTrainedModel.dequantize 恢复到原来的精度，但可能发生一定质量损失。请确保 GPU 有足够显存容纳反量化后的模型。

## semantic_012

**Focus**

Map cross-framework deployment to the export and execution tools.

**考查点（中文）**

将跨框架部署需求对应到导出和执行工具。

**Question**

We need to deploy a model with a runtime that is separate from the original training framework. What export format and supporting library are described for this purpose?

**问题（中文）**

我们需要用独立于原训练框架的运行时部署模型。文档为此介绍了什么导出格式和支持库？

**Reference answer**

Export the model to ONNX, an open standard for representing models through common operators and a computational graph across frameworks. The Optimum library provides export support, and ONNX Runtime can run the exported model through task-specific ORTModel classes.

**参考答案（中文）**

将模型导出为 ONNX。它是一种开放标准，使用通用算子和计算图跨框架表示模型。Optimum 库提供导出支持；ONNX Runtime 可以通过对应任务的 ORTModel 类运行导出的模型。

**Source:** [serialization.md, lines 19–19](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/serialization.md#L19)

**来源：** 同上文档第 19–19 行；以下译文对应所列原文。

**Original evidence**

````text
[ONNX](http://onnx.ai) is an open standard that defines a common set of operators and a file format to represent deep learning models in different frameworks, including PyTorch and TensorFlow. When a model is exported to ONNX, the operators construct a computational graph (or *intermediate representation*) which represents the flow of data through the model. Standardized operators and data types makes it easy to switch between frameworks.
````

**原文依据（中文译文）**

ONNX 是一种开放标准，规定了通用算子和文件格式，用于表示 PyTorch、TensorFlow 等不同框架中的深度学习模型。导出为 ONNX 后，算子构成计算图（或中间表示），描述数据在模型中的流动。标准化的算子和数据类型便于跨框架切换。

**Source:** [serialization.md, lines 21–21](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/serialization.md#L21)

**来源：** 同上文档第 21–21 行；以下译文对应所列原文。

**Original evidence**

````text
The [Optimum](https://huggingface.co/docs/optimum/index) library exports a model to ONNX with configuration objects which are supported for [many architectures](https://huggingface.co/docs/optimum/exporters/onnx/overview) and can be easily extended. If a model isn't supported, feel free to make a [contribution](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute) to Optimum.
````

**原文依据（中文译文）**

Optimum 库利用配置对象将模型导出为 ONNX，支持许多架构，并可方便地扩展。如果某个模型暂不支持，可以向 Optimum 贡献相应实现。

**Source:** [serialization.md, lines 25–27](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/serialization.md#L25)

**来源：** 同上文档第 25–27 行；以下译文对应所列原文。

**Original evidence**

````text
- [Graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) and [quantization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization) for improving inference.
- Use the [`~optimum.onnxruntime.ORTModel`] API to run a model with [ONNX Runtime](https://onnxruntime.ai/).
- Use [optimized inference pipelines](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines) for ONNX models.
````

**原文依据（中文译文）**

导出收益包括：通过计算图优化和量化改善推理；通过 optimum.onnxruntime.ORTModel API 使用 ONNX Runtime 运行模型；为 ONNX 模型使用优化后的推理 pipeline。

## semantic_013

**Focus**

Relate preprocessing throughput to batching and implementation.

**考查点（中文）**

将预处理吞吐量问题对应到批处理和具体实现。

**Question**

Our preprocessing is a bottleneck because we encode text one sentence at a time. How should we organize it, and which tokenizer implementation is especially useful?

**问题（中文）**

我们逐句编码文本，导致预处理成为瓶颈。应该怎样组织预处理？哪种 tokenizer 实现尤其适合？

**Reference answer**

Preprocess batches of text rather than one sentence at a time. Fast tokenizers are Rust-based implementations and are especially effective at parallelizing batched tokenization; they also provide additional alignment methods compared with Python tokenizers.

**参考答案（中文）**

批量预处理文本，而不是每次只处理一句。Fast tokenizer 基于 Rust，尤其擅长并行处理批量分词；相比 Python tokenizer，它还提供额外的对齐方法。

**Source:** [fast_tokenizers.md, lines 311–311](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L311)

**来源：** 同上文档第 311–311 行；以下译文对应所列原文。

**Original evidence**

````text
It is faster and more efficient to preprocess *batches* of text instead of a single sentence at a time. Fast tokenizers are especially good at parallelizing tokenization.
````

**原文依据（中文译文）**

批量预处理文本比每次只处理一句更快、更高效。Fast tokenizer 尤其擅长并行分词。

**Source:** [fast_tokenizers.md, lines 136–136](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L136)

**来源：** 同上文档第 136–136 行；以下译文对应所列原文。

**Original evidence**

````text
[`PreTrainedTokenizerFast`] or *fast tokenizers* are Rust-based tokenizers from the [Tokenizers](https://hf.co/docs/tokenizers) library. It is significantly faster at batched tokenization and provides additional alignment methods compared to the Python-based tokenizers.
````

**原文依据（中文译文）**

PreTrainedTokenizerFast（fast tokenizer）是 Tokenizers 库中基于 Rust 的分词器。它在批量分词时明显更快，并比基于 Python 的 tokenizer 提供更多对齐方法。

## semantic_014

**Focus**

Recover a stopped training workflow.

**考查点（中文）**

恢复中断的训练流程。

**Question**

A training job stopped unexpectedly, but its saved checkpoints are still available. How can we continue from saved progress rather than restarting from scratch?

**问题（中文）**

一个训练任务意外停止，但保存的检查点仍然存在。怎样从已保存的进度继续，而不是从头开始？

**Reference answer**

Call trainer.train with resume_from_checkpoint=True to resume from the last checkpoint, or pass a specific checkpoint path to resume from that checkpoint. Saved checkpoints are intended to support recovery and continuation of training.

**参考答案（中文）**

调用 trainer.train，并设置 resume_from_checkpoint=True，从最后一个检查点恢复；也可以传入具体检查点的路径，从指定位置恢复。保存检查点就是为了支持训练恢复和继续执行。

**Source:** [trainer.md, lines 87–87](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/trainer.md#L87)

**来源：** 同上文档第 87–87 行；以下译文对应所列原文。

**Original evidence**

````text
Saving checkpoints are useful for resuming training or recovering your training progress if you encounter an error. Set the `resume_from_checkpoint` parameter in [`~Trainer.train`] to resume training from the last checkpoint or a specific checkpoint.
````

**原文依据（中文译文）**

保存检查点有助于恢复训练或在出错后恢复训练进度。在 Trainer.train 中设置 resume_from_checkpoint，可以从最后一个检查点或指定检查点恢复。

**Source:** [trainer.md, lines 92–94](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/trainer.md#L92)

**来源：** 同上文档第 92–94 行；以下译文对应所列原文。

**Original evidence**

````text
```py
trainer.train(resume_from_checkpoint=True)
```
````

**原文依据（中文译文）**

代码保持原样：trainer.train(resume_from_checkpoint=True)。这会从最近的检查点恢复训练。

## semantic_015

**Focus**

Interpret task-head initialization as a normal adaptation step.

**考查点（中文）**

将任务头的初始化理解为正常的模型适配步骤。

**Question**

We loaded a general pretrained encoder for review classification and received a warning that the new classification layer has uninitialized weights. Why is this expected, and what should we do?

**问题（中文）**

我们加载了一个通用预训练编码器来做评论分类，随后看到新分类层的权重尚未初始化的警告。为什么这是预期行为？应该怎么办？

**Reference answer**

The pretrained head is replaced by a randomly initialized classification head for the new task. That head must be fine-tuned on the classification dataset before it can produce meaningful predictions. Load the sequence-classification model with the appropriate number of labels and train it.

**参考答案（中文）**

原预训练头被替换为面向新任务、随机初始化的分类头。必须在分类数据集上微调该分类头，才能产生有意义的预测。加载序列分类模型时设置正确的标签数量，并训练模型。

**Source:** [training.md, lines 77–78](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L77)

**来源：** 同上文档第 77–78 行；以下译文对应所列原文。

**Original evidence**

````text
> [!TIP]
> The message above is a reminder that the models pretrained head is discarded and replaced with a randomly initialized classification head. The randomly initialized head needs to be fine-tuned on your specific task to output meaningful predictions.
````

**原文依据（中文译文）**

提示：上述消息说明模型原来的预训练头被丢弃，替换为随机初始化的分类头。需要针对具体任务微调这个头，才能输出有意义的预测。

**Source:** [training.md, lines 72–75](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L72)

**来源：** 同上文档第 72–75 行；以下译文对应所列原文。

**Original evidence**

````text
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
"Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']"
"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
```
````

**原文依据（中文译文）**

示例代码将序列分类模型的 num_labels 设置为 5。日志说明 classifier.bias 和 classifier.weight 并未从原检查点初始化，而是新初始化的；并提醒应先在下游任务上训练模型，再用于预测和推理。代码和参数名保持原样。

## semantic_016

**Focus**

Interpret benchmark results in the context of the actual workload.

**考查点（中文）**

结合实际工作负载解读基准测试结果。

**Question**

We found a published comparison in which one compression method is faster. How should we decide whether that method meets our own deployment requirements?

**问题（中文）**

我们看到一份公开对比，其中某种压缩方法速度更快。如何判断它是否满足我们自己的部署要求？

**Reference answer**

Benchmark the quantized model’s accuracy and speed on your specific task and hardware instead of treating the published comparison as a universal guarantee. Check whether the measured results meet your own requirements and consult the method’s usage documentation.

**参考答案（中文）**

在自己的任务和硬件上测试量化模型的准确率与速度，不要把公开对比当作普遍保证。检查实际测量结果是否符合自己的要求，并查阅该方法的使用文档。

**Source:** [quantization/selecting.md, lines 156–157](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L156)

**来源：** 同上文档第 156–157 行；以下译文对应所列原文。

**Original evidence**

````text
> [!TIP]
> Always benchmark the performance (accuracy and speed) of the quantized model on your specific task and hardware to ensure it meets your requirements. Refer to the individual documentation pages linked above for detailed usage instructions.
````

**原文依据（中文译文）**

提示：始终在自己的任务和硬件上测量量化模型的性能（准确率和速度），确保满足要求。详细用法请参阅上面链接的各方法文档。

## semantic_017

**Focus**

Diagnose long-context state memory rather than weight memory.

**考查点（中文）**

诊断长上下文的状态存储问题，而非权重存储问题。

**Question**

Long conversations are exhausting GPU memory because their accumulated attention state is large. How can we keep less of that state on the GPU, and what performance trade-off follows?

**问题（中文）**

长对话积累的注意力状态很大，导致 GPU 显存耗尽。怎样减少 GPU 上保留的状态？会带来什么性能取舍？

**Reference answer**

Use cache offloading. Most layers’ KV caches are moved to CPU, while the current layer’s cache is maintained on GPU and the next layer is prefetched. This saves GPU memory, but transfers between CPU and GPU can reduce generation throughput.

**参考答案（中文）**

使用缓存卸载。大多数层的 KV 缓存被移到 CPU，GPU 上保留当前层的缓存，并预取下一层的缓存。这能节省显存，但 CPU 与 GPU 之间的数据传输可能降低生成吞吐量。

**Source:** [kv_cache.md, lines 95–95](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L95)

**来源：** 同上文档第 95–95 行；以下译文对应所列原文。

**Original evidence**

````text
Offloading the cache saves GPU memory by moving the KV cache for model layers except one to the CPU. Only the current layer cache is maintained on the GPU during a models `forward` iteration over the layers. It will asynchronously prefetch the next layer's cache, and send back the current layer's cache back to the CPU after attention computation.
````

**原文依据（中文译文）**

缓存卸载将除当前层外的其他层 KV 缓存移到 CPU，从而节省 GPU 显存。在模型 forward 逐层执行时，GPU 只保留当前层的缓存；系统异步预取下一层缓存，并在注意力计算完成后将当前层缓存送回 CPU。

**Source:** [kv_cache.md, lines 99–100](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L99)

**来源：** 同上文档第 99–100 行；以下译文对应所列原文。

**Original evidence**

````text
> [!WARNING]
> You may notice a small degradation in generation throughput compared to a full on-device cache, depending on your model and generation choices (context size, number of generated tokens, number of beams, etc.). This is because moving the key/value states back and forth requires some work.
````

**原文依据（中文译文）**

警告：与完全驻留设备上的缓存相比，生成吞吐量可能略有下降，具体取决于模型和生成配置，例如上下文长度、生成 token 数和束数。原因是键值状态在设备间来回移动需要额外工作。

## semantic_018

**Focus**

Assess workload shape when choosing fixed cache capacity.

**考查点（中文）**

根据工作负载的长度分布评估固定缓存容量。

**Question**

Our generation requests have very different lengths. Why might reserving one large fixed attention-state buffer be inefficient even though it helps compilation?

**问题（中文）**

我们的生成请求长度差异很大。为什么预留一个很大的固定注意力状态缓冲区，即使有助于编译，也可能效率不高？

**Reference answer**

A StaticCache preallocates a maximum size and supports compiling the decoding stage. However, unused positions are masked and still cause wasted attention computation, so a large fixed cache can be inefficient for many short requests mixed with a long request. It is a trade-off that is better suited to similar-length sequences.

**参考答案（中文）**

StaticCache 预分配最大容量，并支持编译解码阶段。但是未使用的位置虽然被掩码屏蔽，仍会造成注意力计算浪费。如果大量短请求与长请求混在一起，大型固定缓存可能效率不高。这种取舍更适合长度相近的序列。

**Source:** [kv_cache.md, lines 72–72](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L72)

**来源：** 同上文档第 72–72 行；以下译文对应所列原文。

**Original evidence**

````text
A fixed-size cache ([`StaticCache`]) pre-allocates a specific maximum cache size for the kv pairs. You can generate up to the maximum cache size without needing to modify it. However, having a fixed (usually large) size for the key/value states means that while generating, a lot of tokens will actually be masked as they should not take part in the attention. So this trick allows to easily `compile` the decoding stage, but it incurs a waste of tokens in the attention computation. As all things, it's then a trade-off which should be very good if you generate with several sequence of more or less the same lengths, but may be sub-optimal if you have for example 1 very large sequence, and then only short sequences (as the fix cache size would be large, a lot would be wasted for the short sequences). Make sure you understand the impact if you use it!
````

**原文依据（中文译文）**

固定大小的 StaticCache 会为 KV 对预分配一个最大缓存容量。在该容量内生成时，不需要改变缓存大小。但较大的固定键值状态中会有许多不应参与注意力的 token 被掩码屏蔽。这样便于编译解码阶段，却会浪费注意力计算。长度相近的多个序列通常适合这种取舍；若一个很长的序列与许多短序列混在一起，大缓存会对短序列造成较多浪费，效果可能不理想。使用前应理解其影响。

## semantic_019

**Focus**

Connect external optimizer installation to the native training interface.

**考查点（中文）**

将外部优化器的安装对应到原生训练接口。

**Question**

We want to use an optimizer supplied by an external library with the standard training loop. What integration steps does the guide describe?

**问题（中文）**

我们想在标准训练循环中使用外部库提供的优化器。文档介绍了哪些接入步骤？

**Reference answer**

Install the library that supplies the optimizer and choose its identifier with the optim parameter in TrainingArguments. Depending on the optimizer, configure additional settings such as optim_target_modules or optim_args as documented, then use those training arguments with Trainer.

**参考答案（中文）**

安装提供优化器的库，并通过 TrainingArguments 的 optim 参数选择其标识符。根据具体优化器，按文档配置 optim_target_modules、optim_args 等额外选项，然后把这些训练参数交给 Trainer。

**Source:** [optimizers.md, lines 19–19](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/optimizers.md#L19)

**来源：** 同上文档第 19–19 行；以下译文对应所列原文。

**Original evidence**

````text
Transformers offers two native optimizers, AdamW and AdaFactor. It also provides integrations for more specialized optimizers. Install the library that offers the optimizer and drop it in the `optim` parameter in [`TrainingArguments`].
````

**原文依据（中文译文）**

Transformers 提供 AdamW 和 AdaFactor 两种原生优化器，也支持接入更专门的优化器。安装提供该优化器的库，并在 TrainingArguments 的 optim 参数中选择它。

**Source:** [optimizers.md, lines 50–50](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/optimizers.md#L50)

**来源：** 同上文档第 50–50 行；以下译文对应所列原文。

**Original evidence**

````text
Use the `optim_target_modules` parameter to specify which layers to train.
````

**原文依据（中文译文）**

使用 optim_target_modules 参数指定需要训练哪些层。

**Source:** [optimizers.md, lines 70–70](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/optimizers.md#L70)

**来源：** 同上文档第 70–70 行；以下译文对应所列原文。

**Original evidence**

````text
For additional training options, use `optim_args` to define hyperparameters like `rank`, `scale`, and more. Refer to the table below for a complete list of available hyperparameters.
````

**原文依据（中文译文）**

如需额外训练选项，使用 optim_args 定义 rank、scale 等超参数。完整参数列表见下表。

## semantic_020

**Focus**

Understand the coupled interfaces behind a custom attention backend.

**考查点（中文）**

理解自定义注意力后端需要配合使用的接口。

**Question**

A custom attention implementation needs a different attention-mask format. What additional registration is needed so the model can construct the appropriate mask?

**问题（中文）**

自定义注意力实现需要不同格式的注意力掩码。还需要注册什么，才能让模型构造正确的掩码？

**Reference answer**

Register a corresponding mask function with AttentionMaskInterface in addition to the custom attention function. The mask interface adapts mask creation to the selected attention implementation. If no matching mask function is registered, mask creation is skipped.

**参考答案（中文）**

除了注册自定义注意力函数，还要通过 AttentionMaskInterface 注册对应的掩码函数。该接口根据所选注意力实现调整掩码创建方式。如果没有注册匹配的掩码函数，就会跳过掩码创建。

**Source:** [attention_interface.md, lines 160–162](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L160)

**来源：** 同上文档第 160–162 行；以下译文对应所列原文。

**Original evidence**

````text
Having a new attention function may mean that you need a new format of attention mask to decide what key and value tokens
the query tokens should attend to. This is now possible with the `AttentionMaskInterface`! It works in the same way as
the `AttentionInterface`:
````

**原文依据（中文译文）**

新的注意力函数可能需要新的掩码格式，以决定查询 token 应关注哪些键和值 token。AttentionMaskInterface 可以实现这一点，工作方式与 AttentionInterface 相同。

**Source:** [attention_interface.md, lines 176–178](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L176)

**来源：** 同上文档第 176–178 行；以下译文对应所列原文。

**Original evidence**

````text
The reason you have to register it is because we need to automatically correct your mask format based on the attention implementation (for example, flex attention uses a BlockMask format, while sdpa uses a 4D tensor).
By default, if you do not register an attention mask function along with your attention function, mask creation will be skipped
and `attention_mask=None` will be passed along to the Attention layers.
````

**原文依据（中文译文）**

必须注册掩码函数，是因为系统需要根据注意力实现自动调整掩码格式，例如 flex attention 使用 BlockMask，而 SDPA 使用四维张量。默认情况下，如果注册注意力函数时没有同时注册掩码函数，会跳过掩码创建，并向注意力层传入 attention_mask=None。

## keyword_001

**Focus**

Exact padding enum behavior.

**考查点（中文）**

准确区分 padding 枚举值的行为。

**Question**

What do padding="longest" and padding="max_length" do when the tokenizer receives a single sequence?

**问题（中文）**

当 tokenizer 只接收一个序列时，padding="longest" 和 padding="max_length" 分别会做什么？

**Reference answer**

padding="longest" (or True) applies no padding to a single sequence. padding="max_length" still pads a single sequence to the specified max_length, or to the model’s maximum length if max_length is not provided.

**参考答案（中文）**

padding="longest"（或 True）不会对单个序列进行填充。padding="max_length" 仍会把单个序列填充到指定的 max_length；如果没有指定 max_length，则填充到模型的最大长度。

**Source:** [pad_truncation.md, lines 25–29](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L25)

**来源：** 同上文档第 25–29 行；以下译文对应所列原文。

**Original evidence**

````text
- `True` or `'longest'`: pad to the longest sequence in the batch (no padding is applied if you only provide
    a single sequence).
- `'max_length'`: pad to a length specified by the `max_length` argument or the maximum length accepted
    by the model if no `max_length` is provided (`max_length=None`). Padding will still be applied if you only provide a single sequence.
- `False` or `'do_not_pad'`: no padding is applied. This is the default behavior.
````

**原文依据（中文译文）**

True 或 "longest"：填充到批次中最长序列的长度；如果只有一个序列，则不填充。"max_length"：填充到指定的 max_length；如果未指定（max_length=None），则使用模型接受的最大长度，单个序列也会填充。False 或 "do_not_pad"：不填充，这是默认行为。

## keyword_002

**Focus**

Precise truncation enum.

**考查点（中文）**

准确定位截断策略的枚举值。

**Question**

For a pair of sequences, which truncation value preserves the first sequence and truncates only the second?

**问题（中文）**

对于一对序列，哪个 truncation 值会保留第一个序列，只截断第二个？

**Reference answer**

Use truncation="only_second". It truncates only the second sequence of a pair to satisfy the specified max_length, or the model’s maximum accepted length when max_length is not provided.

**参考答案（中文）**

使用 truncation="only_second"。它只截断序列对中的第二个序列，使长度符合指定的 max_length；如果未指定，则使用模型允许的最大长度。

**Source:** [pad_truncation.md, lines 33–43](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L33)

**来源：** 同上文档第 33–43 行；以下译文对应所列原文。

**Original evidence**

````text
- `True` or `'longest_first'`: truncate to a maximum length specified by the `max_length` argument or
    the maximum length accepted by the model if no `max_length` is provided (`max_length=None`). This will
    truncate token by token, removing a token from the longest sequence in the pair until the proper length is
    reached.
- `'only_second'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`). This will only truncate
    the second sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `'only_first'`: truncate to a maximum length specified by the `max_length` argument or the maximum
    length accepted by the model if no `max_length` is provided (`max_length=None`). This will only truncate
    the first sentence of a pair if a pair of sequences (or a batch of pairs of sequences) is provided.
- `False` or `'do_not_truncate'`: no truncation is applied. This is the default behavior.
````

**原文依据（中文译文）**

True 或 "longest_first"：截断到指定 max_length，未指定时使用模型的最大长度；在序列对中，每次从较长序列移除一个 token，直到满足长度。"only_second"：只截断序列对中的第二个序列。"only_first"：只截断第一个序列。False 或 "do_not_truncate"：不截断，这是默认行为。

## keyword_003

**Focus**

Locate a task-specific return parameter.

**考查点（中文）**

定位任务专用的返回参数。

**Question**

Which Pipeline parameter value returns only newly generated text instead of including the prompt?

**问题（中文）**

Pipeline 的哪个参数值能只返回新生成的文本，而不包含提示词？

**Reference answer**

Pass return_full_text=False to the text-generation pipeline. It returns the generated continuation rather than the prompt plus generated text.

**参考答案（中文）**

向文本生成 pipeline 传入 return_full_text=False。它会返回生成的续写部分，而不是提示词和生成文本的组合。

**Source:** [pipeline_tutorial.md, lines 256–256](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L256)

**来源：** 同上文档第 256–256 行；以下译文对应所列原文。

**Original evidence**

````text
Pass `return_full_text=False` to [`Pipeline`] to only return the generated text instead of the full text (prompt and generated text).
````

**原文依据（中文译文）**

向 Pipeline 传入 return_full_text=False，只返回生成的文本，而不是完整文本（提示词加生成文本）。

## keyword_004

**Focus**

Exact ASR parameter value.

**考查点（中文）**

准确定位 ASR 参数值。

**Question**

What return_timestamps value requests word-level timestamps from an automatic speech recognition pipeline?

**问题（中文）**

自动语音识别 pipeline 中，return_timestamps 的什么值会请求单词级时间戳？

**Reference answer**

Use return_timestamps="word". It requests timestamps indicating when each word was spoken.

**参考答案（中文）**

使用 return_timestamps="word"，请求每个单词被说出的时间戳。

**Source:** [pipeline_tutorial.md, lines 222–222](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L222)

**来源：** 同上文档第 222–222 行；以下译文对应所列原文。

**Original evidence**

````text
Pass the `return_timestamps="word"` parameter to [`Pipeline`] to return when each word was spoken.
````

**原文依据（中文译文）**

向 Pipeline 传入 return_timestamps="word"，返回每个单词被说出的时间。

## keyword_005

**Focus**

Exact loading parameter semantics.

**考查点（中文）**

准确理解加载参数的语义。

**Question**

What is the documented purpose of dtype="auto" when loading pretrained model weights?

**问题（中文）**

加载预训练权重时，文档中 dtype="auto" 的作用是什么？

**Reference answer**

dtype="auto" initializes weights directly in the data type in which they are stored. This can avoid loading them twice, first in PyTorch’s default float32 and then in the desired data type.

**参考答案（中文）**

dtype="auto" 直接按权重存储时的数据类型初始化权重。这可以避免先以 PyTorch 默认的 float32 加载，再以目标数据类型加载第二次。

**Source:** [quicktour.md, lines 86–87](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quicktour.md#L86)

**来源：** 同上文档第 86–87 行；以下译文对应所列原文。

**Original evidence**

````text
- `device_map="auto"` automatically allocates the model weights to your fastest device first.
- `dtype="auto"` directly initializes the model weights in the data type they're stored in, which can help avoid loading the weights twice (PyTorch loads weights in `torch.float32` by default).
````

**原文依据（中文译文）**

device_map="auto" 优先将权重分配到最快的设备。dtype="auto" 按权重存储时的数据类型直接初始化，可避免加载两次；PyTorch 默认以 torch.float32 加载。

## keyword_006

**Focus**

Exact checkpoint parameter and value.

**考查点（中文）**

准确定位检查点参数及其默认值。

**Question**

What is the documented default max_shard_size in save_pretrained, and what does that parameter control?

**问题（中文）**

save_pretrained 中，文档记录的 max_shard_size 默认值是什么？这个参数控制什么？

**Reference answer**

max_shard_size defaults to 5GB for each checkpoint shard. It controls the shard size used when saving model weights, helping the resulting checkpoint fit memory-limited environments.

**参考答案（中文）**

max_shard_size 默认让每个检查点分片的大小为 5GB。它控制保存模型权重时的分片大小，帮助保存的检查点适用于内存受限的环境。

**Source:** [models.md, lines 119–119](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L119)

**来源：** 同上文档第 119–119 行；以下译文对应所列原文。

**Original evidence**

````text
The `max_shard_size` parameter defaults to 5GB for each shard because it is easier to run on free-tier GPU instances without running out of memory.
````

**原文依据（中文译文）**

max_shard_size 的默认值是每个分片 5GB，便于在免费 GPU 实例上运行而不发生内存不足。

## keyword_007

**Focus**

Framework tensor enum.

**考查点（中文）**

准确定位框架张量的枚举值。

**Question**

Which tokenizer return_tensors value requests PyTorch tensors?

**问题（中文）**

tokenizer 的 return_tensors 取什么值，会请求返回 PyTorch 张量？

**Reference answer**

Set return_tensors="pt" when calling the tokenizer to request PyTorch tensors.

**参考答案（中文）**

调用 tokenizer 时设置 return_tensors="pt"，即可请求 PyTorch 张量。

**Source:** [fast_tokenizers.md, lines 31–37](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L31)

**来源：** 同上文档第 31–37 行；以下译文对应所列原文。

**Original evidence**

````text
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt")
{'input_ids': tensor([[     2,   1734,    708,   1508,   4915,    577,   1500,    692,    573,
         156808, 128149,   9581, 235265]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```
````

**原文依据（中文译文）**

示例代码从 google/gemma-2-2b 加载 tokenizer，并以 return_tensors="pt" 编码句子“我们很高兴向你展示 🤗 Transformers 库”。返回 input_ids 和 attention_mask，二者都是 PyTorch 张量。代码中的键名和数值保持原样。

## keyword_008

**Focus**

Exact flag for two-step chat formatting.

**考查点（中文）**

准确定位两步聊天格式化流程中的开关。

**Question**

After apply_chat_template(tokenize=False), what add_special_tokens value should be used if the formatted string is tokenized later?

**问题（中文）**

调用 apply_chat_template(tokenize=False) 后，如果之后还要对格式化字符串分词，add_special_tokens 应设为什么？

**Reference answer**

Set add_special_tokens=False. The chat template should already contain the required special tokens, so adding them again can duplicate tokens and hurt model performance.

**参考答案（中文）**

设置 add_special_tokens=False。聊天模板应已包含所需特殊 token，再添加一次可能造成重复并影响模型性能。

**Source:** [chat_templating.md, lines 125–127](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L125)

**来源：** 同上文档第 125–127 行；以下译文对应所列原文。

**Original evidence**

````text
> [!WARNING]
> Some tokenizers add special `<bos>` and `<eos>` tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` if you tokenize later to avoid duplicating these tokens.
> This isn't an issue if you use `apply_chat_template(tokenize=True)`, which means it's usually the safer option!
````

**原文依据（中文译文）**

警告：某些 tokenizer 会添加 <bos> 和 <eos> 特殊 token。聊天模板应已包含全部所需特殊 token，再添加可能出错或重复，影响性能。使用 apply_chat_template(tokenize=False) 格式化后，如果之后再分词，应设置 add_special_tokens=False，避免重复。使用 apply_chat_template(tokenize=True) 则没有这个问题，通常更安全。

## keyword_009

**Focus**

Exact Jinja helper names and outputs.

**考查点（中文）**

准确识别 Jinja 辅助函数及其输出。

**Question**

What do raise_exception(msg) and strftime_now(format_str) do inside a chat template?

**问题（中文）**

聊天模板中的 raise_exception(msg) 和 strftime_now(format_str) 分别做什么？

**Reference answer**

raise_exception(msg) raises a TemplateException, useful for debugging or reporting incorrect template usage. strftime_now(format_str) retrieves the current date and time formatted with the supplied format string.

**参考答案（中文）**

raise_exception(msg) 抛出 TemplateException，可用于调试或报告模板使用错误。strftime_now(format_str) 按给定格式字符串返回当前日期和时间。

**Source:** [chat_templating_writing.md, lines 119–120](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating_writing.md#L119)

**来源：** 同上文档第 119–120 行；以下译文对应所列原文。

**Original evidence**

````text
- `raise_exception(msg)` raises a `TemplateException`. This is useful for debugging or warning users about incorrect template usage.
- `strftime_now(format_str)` retrieves the current date and time in a specific format, which is often required in system messages. It is equivalent to [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now) in Python.
````

**原文依据（中文译文）**

raise_exception(msg) 会抛出 TemplateException，可用于调试或提醒模板使用错误。strftime_now(format_str) 按指定格式获取当前日期和时间，系统消息经常需要它；相当于 Python 的 datetime.now().strftime(format_str)。

## keyword_010

**Focus**

Identify the generation-length parameter.

**考查点（中文）**

准确定位生成长度参数。

**Question**

What does max_new_tokens control in text generation?

**问题（中文）**

文本生成中的 max_new_tokens 控制什么？

**Reference answer**

max_new_tokens controls the maximum number of newly generated tokens. Set it explicitly when a longer response is needed instead of relying on a potentially small generation default.

**参考答案（中文）**

max_new_tokens 控制新生成 token 的最大数量。需要更长回答时应显式设置，不要依赖可能较小的默认生成长度。

**Source:** [llm_tutorial.md, lines 150–157](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_tutorial.md#L150)

**来源：** 同上文档第 150–157 行；以下译文对应所列原文。

**Original evidence**

````text
| Option name | Type | Simplified description |
|---|---|---|
| `max_new_tokens` | `int` | Controls the maximum generation length. Be sure to define it, as it usually defaults to a small value. |
| `do_sample` | `bool` | Defines whether generation will sample the next token (`True`), or is greedy instead (`False`). Most use cases should set this flag to `True`. Check [this guide](./generation_strategies) for more information. |
| `temperature` | `float` | How unpredictable the next selected token will be. High values (`>0.8`) are good for creative tasks, low values (e.g. `<0.4`) for tasks that require "thinking". Requires `do_sample=True`. |
| `num_beams` | `int` | When set to `>1`, activates the beam search algorithm. Beam search is good on input-grounded tasks. Check [this guide](./generation_strategies) for more information. |
| `repetition_penalty` | `float` | Set it to `>1.0` if you're seeing the model repeat itself often. Larger values apply a larger penalty. |
| `eos_token_id` | `list[int]` | The token(s) that will cause generation to stop. The default value is usually good, but you can specify a different token. |
````

**原文依据（中文译文）**

选项说明：max_new_tokens（int）控制最大生成长度，默认通常较小，应显式设置；do_sample（bool）决定采样（True）还是贪心生成（False），文档建议大多数场景设为 True；temperature（float）控制下一个 token 的不可预测程度，较高值（>0.8）适合创作，较低值（如 <0.4）适合需要“思考”的任务，要求 do_sample=True；num_beams（int）大于 1 时启用束搜索，适合基于输入的任务；repetition_penalty（float）大于 1.0 时惩罚重复，值越大惩罚越强；eos_token_id（list[int]）指定使生成停止的 token，默认通常合适，也可修改。

## keyword_011

**Focus**

Exact decoding configuration.

**考查点（中文）**

准确定位解码配置。

**Question**

What settings enable multinomial sampling rather than beam search in generate?

**问题（中文）**

在 generate 中，哪些设置启用多项式采样，而不是束搜索？

**Reference answer**

Set do_sample=True and num_beams=1. This samples tokens from the model’s probability distribution without maintaining multiple beams.

**参考答案（中文）**

设置 do_sample=True 和 num_beams=1。这样会从模型的概率分布中采样 token，而不维护多个候选束。

**Source:** [generation_strategies.md, lines 53–53](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_strategies.md#L53)

**来源：** 同上文档第 53–53 行；以下译文对应所列原文。

**Original evidence**

````text
Enable multinomial sampling with `do_sample=True` and `num_beams=1`.
````

**原文依据（中文译文）**

设置 do_sample=True 和 num_beams=1，启用多项式采样。

## keyword_012

**Focus**

Similar cache enum identifiers.

**考查点（中文）**

准确区分相近的缓存枚举名称。

**Question**

Which cache_implementation values enable offloading for dynamic and static caches respectively?

**问题（中文）**

动态缓存和静态缓存分别用哪个 cache_implementation 值启用卸载？

**Reference answer**

Use cache_implementation="offloaded" for the dynamic cache and cache_implementation="offloaded_static" for the static cache. These can be configured in GenerationConfig or generate.

**参考答案（中文）**

动态缓存使用 cache_implementation="offloaded"；静态缓存使用 cache_implementation="offloaded_static"。可以在 GenerationConfig 或 generate 中配置。

**Source:** [kv_cache.md, lines 102–103](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L102)

**来源：** 同上文档第 102–103 行；以下译文对应所列原文。

**Original evidence**

````text
Offloading is available for both [`DynamicCache`] and [`StaticCache`]. You can enable it by configuring `cache_implementation="offloaded"` for the dynamic version, or `cache_implementation="offloaded_static"` for the static version, in either [`GenerationConfig`] or [`~GenerationMixin.generate`].
Additionally, you can also instantiate your own [`DynamicCache`] or [`StaticCache`] with the `offloading=True` option, and pass this cache in `generate` or your model's `forward` (for example, `past_key_values=DynamicCache(config=model.config, offloading=True)` for a dynamic cache).
````

**原文依据（中文译文）**

DynamicCache 和 StaticCache 都支持卸载。在 GenerationConfig 或 generate 中，动态版设置 cache_implementation="offloaded"，静态版设置 cache_implementation="offloaded_static"。也可以直接创建缓存实例并设置 offloading=True，再通过 generate 或模型 forward 传入，例如 past_key_values=DynamicCache(config=model.config, offloading=True)。

## keyword_013

**Focus**

Exact shape and dimension meaning.

**考查点（中文）**

准确理解张量形状和各维度的含义。

**Question**

What attention-mask shape is required when implementing an iterative forward loop with a KV cache?

**问题（中文）**

自己实现带 KV 缓存的迭代 forward 循环时，注意力掩码需要什么形状？

**Reference answer**

The attention mask must have shape (batch_size, past_kv_length + new_tokens_length), covering both previously cached tokens and current tokens.

**参考答案（中文）**

注意力掩码的形状必须是 (batch_size, past_kv_length + new_tokens_length)，同时覆盖之前缓存的 token 和当前 token。

**Source:** [cache_explanation.md, lines 78–78](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/cache_explanation.md#L78)

**来源：** 同上文档第 78–78 行；以下译文对应所列原文。

**Original evidence**

````text
2. When the `forward` method is called iteratively, it's crucial that the attention mask shape matches the combined length of the past and current kv pairs. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is typically handled internally in [`~GenerationMixin.generate`], but if you want to implement your own generation loop with [`Cache`], keep this in mind! The attention mask should hold the past and current token values.
````

**原文依据（中文译文）**

迭代调用 forward 时，注意力掩码形状必须与历史和当前 KV 对的总长度一致，即 (batch_size, past_kv_length + new_tokens_length)。generate 通常会在内部处理；自己实现带 Cache 的生成循环时则需要自行保证，掩码应覆盖历史和当前 token。

## keyword_014

**Focus**

Precise training-evaluation API name.

**考查点（中文）**

准确定位训练评估 API 名称。

**Question**

Which TrainingArguments setting in the classification example reports evaluation accuracy at the end of each epoch?

**问题（中文）**

分类示例中的哪个 TrainingArguments 设置，会在每个 epoch 结束时报告评估准确率？

**Reference answer**

Use eval_strategy="epoch" in TrainingArguments. Supply a compute_metrics function to Trainer to compute the classification metric.

**参考答案（中文）**

在 TrainingArguments 中使用 eval_strategy="epoch"，并向 Trainer 提供 compute_metrics 函数来计算分类指标。

**Source:** [training.md, lines 99–99](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L99)

**来源：** 同上文档第 99–99 行；以下译文对应所列原文。

**Original evidence**

````text
Set up [`TrainingArguments`] with where to save the model and when to compute accuracy during training. The example below sets it to `"epoch"`, which reports the accuracy at the end of each epoch. Add `push_to_hub=True` to upload the model to the Hub after training.
````

**原文依据（中文译文）**

用 TrainingArguments 设置模型保存位置和训练中计算准确率的时机。下面的示例设为 "epoch"，在每个 epoch 结束时报告准确率。添加 push_to_hub=True，可在训练后把模型上传到 Hub。

**Source:** [training.md, lines 113–122](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L113)

**来源：** 同上文档第 113–122 行；以下译文对应所列原文。

**Original evidence**

````text
```py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()
```
````

**原文依据（中文译文）**

代码保持原样：创建 Trainer，传入 model、args、train_dataset、eval_dataset 和 compute_metrics；调用 trainer.train() 开始训练。

## keyword_015

**Focus**

Exact stage-number behavior.

**考查点（中文）**

准确区分阶段编号的行为。

**Question**

What does ZeRO stage 1 shard, and what extra states does ZeRO stage 2 shard?

**问题（中文）**

ZeRO stage 1 对什么进行分片？stage 2 额外对什么状态进行分片？

**Reference answer**

ZeRO-1 shards optimizer states across GPUs. ZeRO-2 shards both optimizer states and gradient states across GPUs.

**参考答案（中文）**

ZeRO-1 在 GPU 之间对优化器状态进行分片；ZeRO-2 同时对优化器状态和梯度状态进行分片。

**Source:** [deepspeed.md, lines 166–166](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/deepspeed.md#L166)

**来源：** 同上文档第 166–166 行；以下译文对应所列原文。

**Original evidence**

````text
ZeRO-1 shards the optimizer states across GPUs and you can expect a small speed up.
````

**原文依据（中文译文）**

ZeRO-1 在 GPU 间对优化器状态分片，可预期获得小幅加速。

**Source:** [deepspeed.md, lines 179–179](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/deepspeed.md#L179)

**来源：** 同上文档第 179–179 行；以下译文对应所列原文。

**Original evidence**

````text
ZeRO-2 shards the optimizer and gradient states across GPUs. This stage is primarily used for training since its features are not relevant to inference. Some important parameters to configure for better performance include the following.
````

**原文依据（中文译文）**

ZeRO-2 在 GPU 间对优化器状态和梯度状态分片。该阶段主要用于训练，因为其功能与推理无关。下面列出了有助于改善性能的重要配置。

## keyword_016

**Focus**

Exact runtime switching method.

**考查点（中文）**

准确定位运行时切换方法。

**Question**

How can set_attn_implementation switch an already loaded model back to SDPA?

**问题（中文）**

如何用 set_attn_implementation 将已加载的模型切换回 SDPA？

**Reference answer**

Call model.set_attn_implementation("sdpa"). This changes the attention function without needing to reload the model.

**参考答案（中文）**

调用 model.set_attn_implementation("sdpa")。它会切换注意力函数，无需重新加载模型。

**Source:** [attention_interface.md, lines 65–67](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L65)

**来源：** 同上文档第 65–67 行；以下译文对应所列原文。

**Original evidence**

````text
```python
# Back to use original sdpa implementation
model.set_attn_implementation("sdpa")
````

**原文依据（中文译文）**

代码保持原样：model.set_attn_implementation("sdpa")。代码注释意为“切换回原来的 SDPA 实现”。

**Source:** [attention_interface.md, lines 72–73](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L72)

**来源：** 同上文档第 72–73 行；以下译文对应所列原文。

**Original evidence**

````text
and it will stop printing the statements, as it now uses the `sdpa` attention.  
This allows to quickly change an attention function, without needing to reload the model!
````

**原文依据（中文译文）**

它会停止打印前述语句，因为现在使用 SDPA 注意力。这可以快速切换注意力函数，而无需重新加载模型。

## keyword_017

**Focus**

Exact nested-quantization flag and documented amount.

**考查点（中文）**

准确定位嵌套量化开关及文档中的节省量。

**Question**

Which BitsAndBytesConfig option enables nested four-bit quantization, and how much extra storage does the guide say it saves?

**问题（中文）**

BitsAndBytesConfig 的哪个选项启用嵌套四位量化？文档称它额外节省多少存储？

**Reference answer**

Set bnb_4bit_use_double_quant=True alongside load_in_4bit=True. Nested quantization quantizes already quantized weights again and saves an additional 0.4 bits per parameter.

**参考答案（中文）**

在 load_in_4bit=True 的同时设置 bnb_4bit_use_double_quant=True。嵌套量化会再次量化已经量化的权重，每个参数额外节省 0.4 bit。

**Source:** [quantization/bitsandbytes.md, lines 319–322](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L319)

**来源：** 同上文档第 319–322 行；以下译文对应所列原文。

**Original evidence**

````text
double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)
````

**原文依据（中文译文）**

代码保持原样：BitsAndBytesConfig 中同时设置 load_in_4bit=True 和 bnb_4bit_use_double_quant=True，构造 double_quant_config。

**Source:** [quantization/bitsandbytes.md, lines 314–314](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L314)

**来源：** 同上文档第 314–314 行；以下译文对应所列原文。

**Original evidence**

````text
Nested quantization can save additional memory at no additional performance cost. This feature performs a second quantization of the already quantized weights to save an additional 0.4 bits/parameter. For example, with nested quantization, you can finetune a [Llama-13b](https://huggingface.co/meta-llama/Llama-2-13b) model on a 16GB NVIDIA T4 GPU with a sequence length of 1024, a batch size of 1, and enable gradient accumulation with 4 steps.
````

**原文依据（中文译文）**

嵌套量化可在不增加性能开销的情况下进一步节省内存。它对已经量化的权重再量化，每个参数额外节省 0.4 bit。例如，可以在 16GB NVIDIA T4 上微调 Llama-13b，序列长度为 1024、批次大小为 1，并使用 4 步梯度累积。

## keyword_018

**Focus**

Exact backend enum and bit-width restriction.

**考查点（中文）**

准确定位后端枚举值和位宽限制。

**Question**

What value of GPTQConfig.backend activates the Marlin kernel, and what bit width does that kernel support?

**问题（中文）**

GPTQConfig.backend 取什么值会启用 Marlin 内核？该内核支持多少位的权重？

**Reference answer**

Set backend="marlin" in GPTQConfig. The documented Marlin kernel supports four-bit GPTQ inference and is optimized for NVIDIA A100 GPUs.

**参考答案（中文）**

在 GPTQConfig 中设置 backend="marlin"。文档介绍的 Marlin 内核支持四位 GPTQ 推理，并针对 NVIDIA A100 GPU 进行了优化。

**Source:** [quantization/gptq.md, lines 121–122](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/gptq.md#L121)

**来源：** 同上文档第 121–122 行；以下译文对应所列原文。

**Original evidence**

````text
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=GPTQConfig(bits=4, backend="marlin"))
```
````

**原文依据（中文译文）**

代码保持原样：加载 GPTQ 模型时传入 GPTQConfig(bits=4, backend="marlin")，并设置 device_map="auto"。

**Source:** [quantization/gptq.md, lines 113–113](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/gptq.md#L113)

**来源：** 同上文档第 113–113 行；以下译文对应所列原文。

**Original evidence**

````text
[Marlin](https://github.com/IST-DASLab/marlin) is a 4-bit only CUDA GPTQ kernel, highly optimized for the NVIDIA A100 GPU (Ampere) architecture. Loading, dequantization, and execution of post-dequantized weights are highly parallelized, offering a substantial inference improvement versus the original CUDA GPTQ kernel. Marlin is only available for quantized inference and does not support model quantization.
````

**原文依据（中文译文）**

Marlin 是仅支持四位的 CUDA GPTQ 内核，针对 NVIDIA A100（Ampere）高度优化。权重加载、反量化及反量化后的执行高度并行，相比原始 CUDA GPTQ 内核能明显改善推理。Marlin 仅用于量化推理，不支持模型量化。

## keyword_019

**Focus**

Exact AWQ fusion controls.

**考查点（中文）**

准确理解 AWQ 融合配置。

**Question**

What do AwqConfig.do_fuse and fuse_max_seq_len configure?

**问题（中文）**

AwqConfig.do_fuse 和 fuse_max_seq_len 分别配置什么？

**Reference answer**

do_fuse=True enables fused AWQ modules. fuse_max_seq_len is the total sequence length, including both the input context and expected generated text; choose a sufficiently large value.

**参考答案（中文）**

do_fuse=True 启用融合的 AWQ 模块。fuse_max_seq_len 是总序列长度，包含输入上下文和预计生成的文本，因此应选择足够大的值。

**Source:** [quantization/awq.md, lines 92–92](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/awq.md#L92)

**来源：** 同上文档第 92–92 行；以下译文对应所列原文。

**Original evidence**

````text
Create an [`AwqConfig`] and set the parameters `fuse_max_seq_len` and `do_fuse=True` to enable fused modules. The `fuse_max_seq_len` parameter is the total sequence length and it should include the context length and the expected generation length. Set it to a larger value to be safe.
````

**原文依据（中文译文）**

创建 AwqConfig，设置 fuse_max_seq_len 和 do_fuse=True，启用融合模块。fuse_max_seq_len 是总序列长度，应包括上下文长度和预计生成长度；为稳妥起见应设置得大一些。

## keyword_020

**Focus**

Locate an optimization-specific tokenizer parameter.

**考查点（中文）**

定位特定优化场景中的 tokenizer 参数。

**Question**

What input-padding control is recommended with static KV cache and torch.compile to reduce shape-related recompilations?

**问题（中文）**

使用静态 KV 缓存和 torch.compile 时，文档推荐用哪个输入填充控制来减少由形状变化造成的重新编译？

**Reference answer**

Left-pad inputs using pad_to_multiple_of to a limited set of values. This reduces changes in input shapes that can trigger recompilation.

**参考答案（中文）**

使用 pad_to_multiple_of 对输入进行左侧填充，让长度落在有限的一组取值上。这样可以减少因输入形状变化而触发的重新编译。

**Source:** [llm_optims.md, lines 40–41](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_optims.md#L40)

**来源：** 同上文档第 40–41 行；以下译文对应所列原文。

**Original evidence**

````text
> [!TIP]
> Regardless of how you use the static kv-cache and torch.compile, left-pad your inputs with [pad_to_multiple_of](https://hf.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of) to a limited set of values to avoid shape-related recompilations.
````

**原文依据（中文译文）**

提示：无论怎样使用静态 KV 缓存和 torch.compile，都应使用 pad_to_multiple_of 将输入左填充到有限的一组长度，避免因形状变化而重新编译。

## distractor_001

**Focus**

Reject conflating padding with truncation.

**考查点（中文）**

识别将填充和截断混为一谈的错误。

**Question**

A developer sets padding=True and claims this alone will shorten overlong inputs to the model limit. Is that correct, and what control is missing?

**问题（中文）**

开发者设置 padding=True，并声称仅靠它就能把过长输入缩短到模型限制以内。这正确吗？缺少什么控制？

**Reference answer**

No. Padding adds padding tokens to shorter sequences; it does not shorten long ones. Enable truncation=True, optionally with max_length, to truncate overlong inputs. Padding and truncation solve opposite length problems.

**参考答案（中文）**

不正确。Padding 给较短序列添加填充 token，不会缩短长序列。应启用 truncation=True，也可以指定 max_length，来截断过长输入。填充和截断处理的是方向相反的长度问题。

**Distractor**

Padding configuration is plausible but does not truncate.

**干扰点（中文）**

填充配置看似相关，但它不会截断输入。

**Source:** [pad_truncation.md, lines 19–19](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L19)

**来源：** 同上文档第 19–19 行；以下译文对应所列原文。

**Original evidence**

````text
Batched inputs are often different lengths, so they can't be converted to fixed-size tensors. Padding and truncation are strategies for dealing with this problem, to create rectangular tensors from batches of varying lengths. Padding adds a special **padding token** to ensure shorter sequences will have the same length as either the longest sequence in a batch or the maximum length accepted by the model. Truncation works in the other direction by truncating long sequences.
````

**原文依据（中文译文）**

批次中的输入长度往往不同，无法直接转换成固定大小的张量。填充和截断用于将这些不同长度的输入组成矩形张量。填充添加特殊 padding token，让较短序列达到批次中的最长长度或模型接受的最大长度；截断则相反，会缩短长序列。

**Source:** [pad_truncation.md, lines 31–31](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L31)

**来源：** 同上文档第 31–31 行；以下译文对应所列原文。

**Original evidence**

````text
The `truncation` argument controls truncation. It can be a boolean or a string:
````

**原文依据（中文译文）**

Truncation 参数控制截断，可以取布尔值或字符串。

## distractor_002

**Focus**

Choose continuation rather than a new-message prefix.

**考查点（中文）**

选择续写已有消息，而不是添加新消息前缀。

**Question**

We want to continue an assistant message that already starts a JSON object. Should we enable add_generation_prompt and continue_final_message together to be safe?

**问题（中文）**

我们想继续生成一条已写出 JSON 对象开头的助手消息。为了稳妥，是否应该同时启用 add_generation_prompt 和 continue_final_message？

**Reference answer**

No. Use continue_final_message=True to continue the existing final message. add_generation_prompt starts a new assistant message, whereas continue_final_message removes end-of-sequence markers to continue one. Combining the two returns an error.

**参考答案（中文）**

不应该。使用 continue_final_message=True，继续现有的最后一条消息。add_generation_prompt 会开始一条新助手消息；continue_final_message 则移除序列结束标记，以续写现有消息。同时使用两者会报错。

**Distractor**

Both flags sound relevant, but their message-boundary operations conflict.

**干扰点（中文）**

两个开关看似都相关，但它们对消息边界的操作互相冲突。

**Source:** [chat_templating.md, lines 177–177](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L177)

**来源：** 同上文档第 177–177 行；以下译文对应所列原文。

**Original evidence**

````text
This is useful for “prefilling” a model response. In the example below, the model generates text that continues the JSON string rather than starting a new message. It can be very useful for improving the accuracy of instruction following when you know how to start its replies.
````

**原文依据（中文译文）**

这对“预填充”模型回复很有用。下面的例子让模型继续已有 JSON 字符串，而不是开始新消息。在已知回复开头时，这能帮助提高指令遵循的准确性。

**Source:** [chat_templating.md, lines 189–190](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L189)

**来源：** 同上文档第 189–190 行；以下译文对应所列原文。

**Original evidence**

````text
> [!WARNING]
> You shouldn't use [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) and [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) together. The former adds tokens that start a new message, while the latter removes end of sequence tokens. Using them together returns an error.
````

**原文依据（中文译文）**

警告：不要同时使用 add_generation_prompt 和 continue_final_message。前者添加开始新消息的 token，后者移除序列结束 token；同时使用会报错。

## distractor_003

**Focus**

Distinguish training formatting from inference prompting.

**考查点（中文）**

区分训练格式化和推理提示。

**Question**

For supervised training on completed conversations, should we keep add_generation_prompt=True because it is helpful when generating assistant replies?

**问题（中文）**

对已经完成的对话进行监督训练时，是否应该保留 add_generation_prompt=True，因为它在生成助手回复时很有用？

**Reference answer**

No. Apply the chat template while preprocessing training data, but set add_generation_prompt=False. The extra tokens that prompt a new assistant response are not helpful during training on the supplied conversations.

**参考答案（中文）**

不应该。预处理训练数据时应用聊天模板，但设置 add_generation_prompt=False。在已有对话上训练时，提示模型开始新助手回复的额外 token 并无帮助。

**Distractor**

An inference setting is incorrectly transferred to training.

**干扰点（中文）**

错误地把推理设置直接用于训练。

**Source:** [chat_templating.md, lines 196–196](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L196)

**来源：** 同上文档第 196–196 行；以下译文对应所列原文。

**Original evidence**

````text
Training a model with a chat template is a good way to ensure the template matches the tokens the model was trained on. Apply the chat template as a preprocessing step to your dataset. Set `add_generation_prompt=False` because the additional tokens to prompt an assistant response aren't helpful during training.
````

**原文依据（中文译文）**

使用聊天模板训练，可以让模板匹配训练时的 token。将模板作为数据集预处理步骤，设置 add_generation_prompt=False，因为提示助手开始新回复的额外 token 对训练没有帮助。

## distractor_004

**Focus**

Distinguish local token selection from sequence-level candidates.

**考查点（中文）**

区分局部 token 选择和序列级候选。

**Question**

Is beam search the same as greedily choosing the highest-probability next token at every step? Explain the difference.

**问题（中文）**

束搜索是否等同于每一步都贪心地选择概率最高的下一个 token？请解释区别。

**Reference answer**

No. Greedy search selects the most likely next token at each step. Beam search tracks several candidate sequences and selects one with the highest overall probability, so it can retain a sequence whose early tokens are not locally the most likely.

**参考答案（中文）**

不等同。贪心搜索在每一步选择最可能的下一个 token；束搜索维护多个候选序列，并选择整体概率最高的序列，因此可以保留早期 token 并非局部最优的序列。

**Distractor**

Both pursue probability, but optimize over different candidate structures.

**干扰点（中文）**

两者都考虑概率，但维护的候选结构不同。

**Source:** [generation_strategies.md, lines 29–29](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_strategies.md#L29)

**来源：** 同上文档第 29–29 行；以下译文对应所列原文。

**Original evidence**

````text
Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in [`GenerationConfig`], this strategy generates a maximum of 20 new tokens.
````

**原文依据（中文译文）**

贪心搜索是默认解码策略，每一步选择最可能的下一个 token。除非在 GenerationConfig 中指定，否则最多生成 20 个新 token。

**Source:** [generation_strategies.md, lines 73–73](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_strategies.md#L73)

**来源：** 同上文档第 73–73 行；以下译文对应所列原文。

**Original evidence**

````text
Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest *overall* probability. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. It is best suited for input-grounded tasks, like describing an image or speech recognition. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps.
````

**原文依据（中文译文）**

束搜索在每一步维护多个生成序列（束），之后选择整体概率最高的序列。与贪心搜索不同，它能“向前看”，保留初始 token 概率较低但整体概率更高的序列，适用于图像描述、语音识别等基于输入的任务。也可设置 do_sample=True，在每一步采样，但各步之间仍会贪心地剪除低概率序列。

## distractor_005

**Focus**

Respect matched generation/detection settings.

**考查点（中文）**

要求生成和检测配置匹配。

**Question**

Can WatermarkDetector use an arbitrary WatermarkingConfig to detect text that was generated with another configuration?

**问题（中文）**

WatermarkDetector 能否使用任意 WatermarkingConfig，检测使用另一配置生成的文本？

**Reference answer**

No. WatermarkDetector must use the same WatermarkingConfig as the one used during generation. It detects the proportion of green tokens under that watermarking scheme.

**参考答案（中文）**

不能。WatermarkDetector 必须使用与生成时相同的 WatermarkingConfig。它会检测该水印方案下绿色 token 的比例。

**Distractor**

A detector with the right class name but wrong configuration is not equivalent.

**干扰点（中文）**

检测器类名正确，不代表错误的配置也能等效工作。

**Source:** [generation_features.md, lines 75–75](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L75)

**来源：** 同上文档第 75–75 行；以下译文对应所列原文。

**Original evidence**

````text
Create an instance of [`WatermarkDetector`] and pass the model output to it to detect whether the text is machine-generated. The [`WatermarkDetector`] must have the same [`WatermarkingConfig`] used during generation.
````

**原文依据（中文译文）**

创建 WatermarkDetector 实例，把模型输出传给它，检测是否由机器生成。检测器必须使用生成时相同的 WatermarkingConfig。

**Source:** [generation_features.md, lines 57–58](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L57)

**来源：** 同上文档第 57–58 行；以下译文对应所列原文。

**Original evidence**

````text
> [!TIP]
> The [`WatermarkDetector`] class detects the proportion of green tokens in generated text, which is why it is recommended to strip the prompt text, if it is much longer than the generated text. Padding can also have an effect on [`WatermarkDetector`].
````

**原文依据（中文译文）**

提示：WatermarkDetector 检测生成文本中绿色 token 的比例。如果提示词远长于生成文本，建议去掉提示词。Padding 也可能影响检测器。

## distractor_006

**Focus**

Distinguish architecture configuration from trained checkpoint weights.

**考查点（中文）**

区分架构配置和已训练的检查点权重。

**Question**

Does constructing a model from its configuration give us the pretrained checkpoint’s learned weights?

**问题（中文）**

仅根据配置构造模型，会获得预训练检查点中已学习的权重吗？

**Reference answer**

No. A configuration describes the architecture and its attributes; constructing the model this way yields random weights that need training. Load learned checkpoint weights with from_pretrained instead.

**参考答案（中文）**

不会。配置描述架构及其属性；这样构造出来的模型具有随机权重，需要训练。应使用 from_pretrained 加载已经学习好的检查点权重。

**Distractor**

Architecture compatibility is mistaken for pretrained state.

**干扰点（中文）**

把架构兼容误认为已经包含预训练状态。

**Source:** [models.md, lines 36–36](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L36)

**来源：** 同上文档第 36–36 行；以下译文对应所列原文。

**Original evidence**

````text
All models have a `configuration.py` file with specific attributes like the number of hidden layers, vocabulary size, activation function, and more. You'll also find a `modeling.py` file that defines the layers and mathematical operations taking place inside each layer. The `modeling.py` file takes the model attributes in `configuration.py` and builds the model accordingly. At this point, you have a model with random weights that needs to be trained to output meaningful results.
````

**原文依据（中文译文）**

每个模型都有 configuration.py，记录隐藏层数量、词表大小、激活函数等属性；modeling.py 则定义各层及层内的数学运算。modeling.py 根据 configuration.py 中的属性构造模型。此时模型的权重是随机的，需要训练才能产生有意义的结果。

**Source:** [models.md, lines 50–50](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L50)

**来源：** 同上文档第 50–50 行；以下译文对应所列原文。

**Original evidence**

````text
To get a pretrained model, you need to load the weights into the model. This is done by calling [`~PreTrainedModel.from_pretrained`] which accepts weights from the Hugging Face Hub or a local directory.
````

**原文依据（中文译文）**

要获得预训练模型，需要把权重加载进模型。通过 PreTrainedModel.from_pretrained 可以加载 Hugging Face Hub 或本地目录中的权重。

## distractor_007

**Focus**

Disambiguate token IDs from the model-specific vocabulary they index.

**考查点（中文）**

区分 token ID 与它索引的模型专用词表。

**Question**

Can we freely replace a pretrained model’s tokenizer with another tokenizer just because both produce input IDs?

**问题（中文）**

只要两个 tokenizer 都输出 input IDs，就能随意替换预训练模型原来的 tokenizer 吗？

**Reference answer**

No. The tokenizer’s vocabulary must match the vocabulary expected by the pretrained model. A custom tokenizer with a different vocabulary cannot be assumed interchangeable merely because it produces numeric input IDs.

**参考答案（中文）**

不能。Tokenizer 的词表必须匹配预训练模型所期望的词表。不能仅因为自定义 tokenizer 输出数值 ID，就认为不同词表的 tokenizer 可以互换。

**Distractor**

Identical output type does not imply matching token semantics.

**干扰点（中文）**

输出类型相同，不代表 token 的语义相同。

**Source:** [fast_tokenizers.md, lines 39–39](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L39)

**来源：** 同上文档第 39–39 行；以下译文对应所列原文。

**Original evidence**

````text
Whichever tokenizer you use, make sure the tokenizer vocabulary is the same as the pretrained models tokenizer vocabulary. This is especially important if you're using a custom tokenizer with a different vocabulary from the pretrained models tokenizer.
````

**原文依据（中文译文）**

无论使用哪种 tokenizer，都要确保词表与预训练模型的 tokenizer 词表相同。使用词表不同的自定义 tokenizer 时尤其要注意。

## distractor_008

**Focus**

Distinguish selecting one adapter from enabling adapters broadly.

**考查点（中文）**

区分选中一个适配器和广泛启用适配器。

**Question**

We have two attached adapters. Does enable_adapters select only the second adapter in the same way as set_adapter("lora_2")?

**问题（中文）**

模型上有两个适配器。enable_adapters 是否和 set_adapter("lora_2") 一样，只选择第二个适配器？

**Reference answer**

No. set_adapter("lora_2") selects that adapter and disables the others. enable_adapters is a broader operation that enables all adapters attached to the model.

**参考答案（中文）**

不是。set_adapter("lora_2") 选择该适配器并禁用其他适配器；enable_adapters 是更广泛的操作，会启用模型上所有已附加的适配器。

**Distractor**

Similar method names have different selection scope.

**干扰点（中文）**

相似的方法名具有不同的选择范围。

**Source:** [peft.md, lines 136–136](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/peft.md#L136)

**来源：** 同上文档第 136–136 行；以下译文对应所列原文。

**Original evidence**

````text
Once added, use [`~integrations.PeftAdapterMixin.set_adapter`] to force a model to use the specified adapter and disable the other adapters.
````

**原文依据（中文译文）**

添加适配器后，使用 PeftAdapterMixin.set_adapter 强制模型使用指定适配器，并禁用其他适配器。

**Source:** [peft.md, lines 144–144](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/peft.md#L144)

**来源：** 同上文档第 144–144 行；以下译文对应所列原文。

**Original evidence**

````text
[`~integrations.PeftAdapterMixin.enable_adapters`] is a broader function that enables *all* adapters attached to a model, and [`~integrations.PeftAdapterMixin.disable_adapters`] disables *all* attached adapters.
````

**原文依据（中文译文）**

PeftAdapterMixin.enable_adapters 会启用模型上全部已附加适配器；disable_adapters 则禁用全部已附加适配器。

## distractor_009

**Focus**

Reject moving an inference cache optimization into training.

**考查点（中文）**

避免把推理缓存优化用于训练。

**Question**

Because cached attention state speeds up generation, should we also enable it during model training?

**问题（中文）**

缓存注意力状态可以加速生成，所以是否也应该在模型训练时启用它？

**Reference answer**

No. The documentation says caching should be used only for inference and may cause unexpected errors if enabled during training.

**参考答案（中文）**

不应该。文档明确说明缓存只能用于推理，在训练时启用可能导致意外错误。

**Distractor**

A generation speed optimization is wrongly applied to training.

**干扰点（中文）**

把生成速度优化错误地用于训练。

**Source:** [cache_explanation.md, lines 29–30](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/cache_explanation.md#L29)

**来源：** 同上文档第 29–30 行；以下译文对应所列原文。

**Original evidence**

````text
> [!WARNING]
> Caching should only be used for **inference**. It may cause unexpected errors if it's enabled during training.
````

**原文依据（中文译文）**

警告：缓存只能用于推理，在训练时启用可能导致意外错误。

## distractor_010

**Focus**

Distinguish cache positions from a count of unpadded tokens.

**考查点（中文）**

区分缓存位置和未填充 token 的数量。

**Question**

A reused KV cache contains ten tokens, including padding positions. Should cache_position for the next token subtract the padding count?

**问题（中文）**

复用的 KV 缓存包含十个 token，其中包括填充位置。下一个 token 的 cache_position 是否应该减去填充数量？

**Reference answer**

No. cache_position is unaffected by padding and advances once per token. For a cache containing ten tokens, the next position is torch.tensor([10]), regardless of padding tokens.

**参考答案（中文）**

不应该。cache_position 不受填充影响，每个 token 都会让它增加一个位置。缓存包含十个 token 时，下一个位置是 torch.tensor([10])，与其中是否有填充 token 无关。

**Distractor**

Masking padding does not change the positional counter for cached tokens.

**干扰点（中文）**

对填充进行掩码屏蔽，不会改变缓存 token 的位置计数。

**Source:** [cache_explanation.md, lines 80–80](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/cache_explanation.md#L80)

**来源：** 同上文档第 80–80 行；以下译文对应所列原文。

**Original evidence**

````text
3. It is also important to be aware of the `cache_position`. This is important if you want to reuse a prefilled [`Cache`] with the `forward` method because you have to pass a valid `cache_position` value. This indicates the input positions in a sequence. `cache_position` is unaffected by padding, and it always adds one more position for each token. For example, if a kv cache contains 10 tokens - regardless of pad tokens - the cache position for the next token should be `torch.tensor([10])`.
````

**原文依据（中文译文）**

复用预填充的 Cache 并调用 forward 时，必须传入有效的 cache_position，表示输入在序列中的位置。cache_position 不受填充影响，每个 token 都使位置增加 1。例如 KV 缓存包含 10 个 token，不论是否有 padding，下一 token 的位置应为 torch.tensor([10])。

## distractor_011

**Focus**

Distinguish dynamic growth from compile-friendly fixed capacity.

**考查点（中文）**

区分动态增长和便于编译的固定容量。

**Question**

Is the dynamically growing cache the preferred choice when we specifically need a fixed shape for compiling decoding?

**问题（中文）**

如果我们明确需要固定形状来编译解码过程，动态增长的缓存是不是首选？

**Reference answer**

No. DynamicCache grows as generation progresses. StaticCache preallocates a maximum capacity and supports compiling decoding, at the cost of unused masked positions and potentially wasted computation.

**参考答案（中文）**

不是。DynamicCache 随生成过程增长；StaticCache 预分配最大容量并支持编译解码，但未使用的掩码位置可能造成计算浪费。

**Distractor**

Dynamic storage flexibility is confused with static-shape compilation.

**干扰点（中文）**

把动态存储的灵活性与静态形状编译混淆。

**Source:** [kv_cache.md, lines 35–35](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L35)

**来源：** 同上文档第 35–35 行；以下译文对应所列原文。

**Original evidence**

````text
The [`DynamicCache`] is the default cache class for all models. It allows the cache size to grow dynamically in order to store an increasing number of keys and values as generation progresses.
````

**原文依据（中文译文）**

DynamicCache 是所有模型的默认缓存类。它允许缓存大小随着生成过程动态增长，存储不断增加的键和值。

**Source:** [kv_cache.md, lines 72–72](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L72)

**来源：** 同上文档第 72–72 行；以下译文对应所列原文。

**Original evidence**

````text
A fixed-size cache ([`StaticCache`]) pre-allocates a specific maximum cache size for the kv pairs. You can generate up to the maximum cache size without needing to modify it. However, having a fixed (usually large) size for the key/value states means that while generating, a lot of tokens will actually be masked as they should not take part in the attention. So this trick allows to easily `compile` the decoding stage, but it incurs a waste of tokens in the attention computation. As all things, it's then a trade-off which should be very good if you generate with several sequence of more or less the same lengths, but may be sub-optimal if you have for example 1 very large sequence, and then only short sequences (as the fix cache size would be large, a lot would be wasted for the short sequences). Make sure you understand the impact if you use it!
````

**原文依据（中文译文）**

固定大小的 StaticCache 会为 KV 对预分配一个最大缓存容量。在该容量内生成时，不需要改变缓存大小。但较大的固定键值状态中会有许多不应参与注意力的 token 被掩码屏蔽。这样便于编译解码阶段，却会浪费注意力计算。长度相近的多个序列通常适合这种取舍；若一个很长的序列与许多短序列混在一起，大缓存会对短序列造成较多浪费，效果可能不理想。使用前应理解其影响。

## distractor_012

**Focus**

Identify exceptions to cache/graph reuse.

**考查点（中文）**

识别缓存和计算图复用的例外条件。

**Question**

If we enable a static KV cache and compile the model, can we freely change batch size or increase maximum output length without triggering recompilation?

**问题（中文）**

启用静态 KV 缓存并编译模型后，能否任意改变批次大小或增大最大输出长度，而不触发重新编译？

**Reference answer**

No. Changing batch size or increasing maximum output length between calls can reinitialize and recompile the cache. generate tries to reuse the cache to avoid recompilation, so workload shape changes still matter.

**参考答案（中文）**

不能。调用之间改变批次大小或增大最大输出长度，可能让缓存重新初始化并重新编译。generate 会尝试复用缓存来避免重新编译，因此工作负载的形状变化仍然重要。

**Distractor**

A static cache does not guarantee reuse across arbitrary shape changes.

**干扰点（中文）**

静态缓存并不保证在任意形状变化后仍能复用。

**Source:** [llm_optims.md, lines 71–72](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_optims.md#L71)

**来源：** 同上文档第 71–72 行；以下译文对应所列原文。

**Original evidence**

````text
1. If the batch size changes or the maximum output length increases between calls, the cache is reinitialized and recompiled.
2. The first several calls of the compiled function are slower because it is being compiled.
````

**原文依据（中文译文）**

如果调用之间改变批次大小或增大最大输出长度，缓存会重新初始化并重新编译。编译函数最初的几次调用较慢，因为正在进行编译。

**Source:** [llm_optims.md, lines 69–69](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_optims.md#L69)

**来源：** 同上文档第 69–69 行；以下译文对应所列原文。

**Original evidence**

````text
Under the hood, [`~GenerationMixin.generate`] attempts to reuse the same cache object to avoid recompilation at each call, which is critical to get the most out of [torch.compile](./perf_torch_compile). Be aware of the following to avoid triggering recompilation or if generation is slower than expected.
````

**原文依据（中文译文）**

generate 在内部尝试复用同一个缓存对象，避免每次调用都重新编译，这对发挥 torch.compile 的效果很重要。应注意下面的情况，避免触发重新编译或出现生成速度不及预期的问题。

## distractor_013

**Focus**

Identify a feature-triggered attention backend fallback.

**考查点（中文）**

识别功能选项触发的注意力后端回退。

**Question**

We chose SDPA but also request output_attentions=True. Should we assume the same fast attention implementation is still being used?

**问题（中文）**

我们选择了 SDPA，同时请求 output_attentions=True。是否可以认为仍在使用同一个快速注意力实现？

**Reference answer**

No. The guide lists output_attentions=True and head_mask as unsupported by SDPA and says Transformers warns and falls back to the slower eager implementation for such options.

**参考答案（中文）**

不可以。文档将 output_attentions=True 和 head_mask 列为 SDPA 不支持的选项，并说明 Transformers 会发出警告，退回较慢的 eager 实现。

**Distractor**

The configured backend name does not guarantee it remains active with unsupported options.

**干扰点（中文）**

配置的后端名称不能保证在使用不支持的选项时仍然生效。

**Source:** [perf_infer_gpu_one.md, lines 178–178](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_infer_gpu_one.md#L178)

**来源：** 同上文档第 178–178 行；以下译文对应所列原文。

**Original evidence**

````text
SDPA is used by default for PyTorch v2.1.1. and greater when an implementation is available. You could explicitly enable SDPA by setting `attn_implementation="sdpa"` in [`~PreTrainedModel.from_pretrained`] though. Certain attention parameters, such as `head_mask` and `output_attentions=True`, are unsupported and returns a warning that Transformers will fall back to the (slower) eager implementation.
````

**原文依据（中文译文）**

在 PyTorch 2.1.1 及以上版本，如果有对应实现，默认使用 SDPA；也可以在 from_pretrained 中显式设置 attn_implementation="sdpa"。head_mask、output_attentions=True 等参数不被支持，此时 Transformers 会警告并退回较慢的 eager 实现。

## distractor_014

**Focus**

Reject an explicitly incompatible optimization combination.

**考查点（中文）**

识别文档明确指出的不兼容优化组合。

**Question**

Can we combine fused AWQ modules with FlashAttention2 to stack both optimizations?

**问题（中文）**

能否把融合的 AWQ 模块和 FlashAttention2 组合起来，叠加两种优化？

**Reference answer**

No. The AWQ documentation explicitly says fused modules cannot be combined with optimization techniques such as FlashAttention2. Choose a compatible configuration instead of enabling both together.

**参考答案（中文）**

不能。AWQ 文档明确说明融合模块不能与 FlashAttention2 等优化技术组合。应选择兼容配置，而不是同时启用两者。

**Distractor**

Two individually useful optimizations are assumed compatible.

**干扰点（中文）**

误以为两种各自有用的优化一定可以组合。

**Source:** [quantization/awq.md, lines 86–87](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/awq.md#L86)

**来源：** 同上文档第 86–87 行；以下译文对应所列原文。

**Original evidence**

````text
> [!WARNING]
> Fused modules cannot be combined with other optimization techniques such as FlashAttention2.
````

**原文依据（中文译文）**

警告：融合模块不能与 FlashAttention2 等其他优化技术组合使用。

## distractor_015

**Focus**

Separate quantization creation from quantized execution.

**考查点（中文）**

区分创建量化模型和执行量化模型。

**Question**

Can the Marlin backend perform the initial GPTQ quantization of a full-precision model, as well as accelerate its later inference?

**问题（中文）**

Marlin 后端能否既负责将全精度模型初次量化为 GPTQ，又加速后续推理？

**Reference answer**

No. The documented Marlin backend is for quantized inference only and does not support model quantization. Initial quantization must be performed by a supported quantization workflow before Marlin is used for inference.

**参考答案（中文）**

不能。文档中的 Marlin 后端仅用于量化模型的推理，不支持模型量化。应先通过支持的量化流程完成初次量化，再使用 Marlin 推理。

**Distractor**

An inference kernel is incorrectly treated as a quantizer.

**干扰点（中文）**

把推理内核误认为量化工具。

**Source:** [quantization/gptq.md, lines 113–113](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/gptq.md#L113)

**来源：** 同上文档第 113–113 行；以下译文对应所列原文。

**Original evidence**

````text
[Marlin](https://github.com/IST-DASLab/marlin) is a 4-bit only CUDA GPTQ kernel, highly optimized for the NVIDIA A100 GPU (Ampere) architecture. Loading, dequantization, and execution of post-dequantized weights are highly parallelized, offering a substantial inference improvement versus the original CUDA GPTQ kernel. Marlin is only available for quantized inference and does not support model quantization.
````

**原文依据（中文译文）**

Marlin 是仅支持四位的 CUDA GPTQ 内核，针对 NVIDIA A100（Ampere）高度优化。权重加载、反量化及反量化后的执行高度并行，相比原始 CUDA GPTQ 内核能明显改善推理。Marlin 仅用于量化推理，不支持模型量化。

## distractor_016

**Focus**

Separate quantized GPU loading from CPU offload representation.

**考查点（中文）**

区分 GPU 量化加载和 CPU 卸载后的表示。

**Question**

After enabling eight-bit CPU offloading, are the offloaded CPU weights also stored as eight-bit integers?

**问题（中文）**

启用八位 CPU 卸载后，被卸载到 CPU 的权重也以八位整数存储吗？

**Reference answer**

No. The bitsandbytes offloading section says weights dispatched to CPU are stored in float32 and are not converted to eight-bit.

**参考答案（中文）**

不是。Bitsandbytes 的卸载章节说明，分配到 CPU 的权重以 float32 存储，不会转换成八位。

**Distractor**

The model’s eight-bit label is incorrectly applied to every device’s storage.

**干扰点（中文）**

错误地把模型的“八位”标签套用到所有设备的存储。

**Source:** [quantization/bitsandbytes.md, lines 200–200](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L200)

**来源：** 同上文档第 200–200 行；以下译文对应所列原文。

**Original evidence**

````text
8-bit models can offload weights between the CPU and GPU to fit very large models into memory. The weights dispatched to the CPU are stored in **float32** and aren't converted to 8-bit. For example, enable offloading for [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7) through [`BitsAndBytesConfig`].
````

**原文依据（中文译文）**

八位模型可在 CPU 与 GPU 之间卸载权重，以容纳更大的模型。分配到 CPU 的权重以 float32 存储，不会转成八位。可以通过 BitsAndBytesConfig 为 bigscience/bloom-1b7 等模型启用卸载。

## distractor_017

**Focus**

Separate quantized weight representation from computation dtype.

**考查点（中文）**

区分量化权重的表示和计算数据类型。

**Question**

Do bnb_4bit_quant_type="nf4" and bnb_4bit_compute_dtype=torch.bfloat16 control the same thing?

**问题（中文）**

bnb_4bit_quant_type="nf4" 和 bnb_4bit_compute_dtype=torch.bfloat16 控制的是同一件事吗？

**Reference answer**

No. bnb_4bit_quant_type="nf4" selects the four-bit weight quantization representation, which the guide recommends for normally distributed weights when training four-bit base models. bnb_4bit_compute_dtype selects the computation data type, such as bfloat16; it does not select the four-bit storage format.

**参考答案（中文）**

不是。bnb_4bit_quant_type="nf4" 选择四位权重的量化表示，文档推荐在训练具有正态分布权重的四位基础模型时使用它。bnb_4bit_compute_dtype 选择计算时的数据类型，例如 bfloat16，而不是四位存储格式。

**Distractor**

Similar quantization settings operate on different kinds of precision.

**干扰点（中文）**

看似相似的量化设置作用于不同层面的精度。

**Source:** [quantization/bitsandbytes.md, lines 284–284](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L284)

**来源：** 同上文档第 284–284 行；以下译文对应所列原文。

**Original evidence**

````text
### Compute data type
````

**原文依据（中文译文）**

计算数据类型。

**Source:** [quantization/bitsandbytes.md, lines 297–297](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L297)

**来源：** 同上文档第 297–297 行；以下译文对应所列原文。

**Original evidence**

````text
NF4 is a 4-bit data type from the [QLoRA](https://hf.co/papers/2305.14314) paper, adapted for weights initialized from a normal distribution. You should use NF4 for training 4-bit base models.
````

**原文依据（中文译文）**

NF4 是 QLoRA 论文提出的四位数据类型，适用于从正态分布初始化的权重。训练四位基础模型时应使用 NF4。

**Source:** [quantization/bitsandbytes.md, lines 302–305](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L302)

**来源：** 同上文档第 302–305 行；以下译文对应所列原文。

**Original evidence**

````text
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
````

**原文依据（中文译文）**

代码保持原样：nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")。

## distractor_018

**Focus**

Distinguish calibration-based methods from on-the-fly methods.

**考查点（中文）**

区分基于校准的方法和即时量化方法。

**Question**

Are GPTQ and AWQ on-the-fly methods that require no calibration dataset, just like the no-calibration approaches listed in the guide?

**问题（中文）**

GPTQ 和 AWQ 是否和文档列出的无需校准方法一样，能即时量化而不需要校准数据集？

**Reference answer**

No. The selecting guide groups GPTQ/GPTQModel and AWQ under Calibration-based Quantization, which requires an upfront calibration step using a dataset. bitsandbytes, HQQ and torchao are listed under No Calibration Required.

**参考答案（中文）**

不是。选择指南把 GPTQ/GPTQModel 和 AWQ 归为基于校准的量化，需要先用数据集进行校准。Bitsandbytes、HQQ 和 torchao 则列在无需校准的方法中。

**Distractor**

All reduced-precision loading workflows are assumed to have the same preparation needs.

**干扰点（中文）**

误以为所有低精度加载流程都有相同的准备要求。

**Source:** [quantization/selecting.md, lines 35–35](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L35)

**来源：** 同上文档第 35–35 行；以下译文对应所列原文。

**Original evidence**

````text
### No Calibration Required (On-the-fly Quantization)
````

**原文依据（中文译文）**

无需校准（即时量化）。

**Source:** [quantization/selecting.md, lines 71–71](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L71)

**来源：** 同上文档第 71–71 行；以下译文对应所列原文。

**Original evidence**

````text
These methods require an upfront calibration step using a dataset to potentially achieve higher accuracy.
````

**原文依据（中文译文）**

这些方法需要先使用数据集进行校准，可能获得更高准确率。

**Source:** [quantization/selecting.md, lines 73–73](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L73)

**来源：** 同上文档第 73–73 行；以下译文对应所列原文。

**Original evidence**

````text
#### GPTQ/GPTQModel
````

**原文依据（中文译文）**

GPTQ/GPTQModel（方法名称保持原样）。

**Source:** [quantization/selecting.md, lines 85–85](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L85)

**来源：** 同上文档第 85–85 行；以下译文对应所列原文。

**Original evidence**

````text
#### AWQ (Activation-aware Weight Quantization)
````

**原文依据（中文译文）**

AWQ（感知激活的权重量化）。

## distractor_019

**Focus**

Separate batch throughput opportunities from live latency constraints.

**考查点（中文）**

区分批处理的吞吐量收益和实时延迟约束。

**Question**

We serve individual low-latency requests on a CPU. Should we increase Pipeline batch_size because batching is guaranteed to make inference faster?

**问题（中文）**

我们在 CPU 上处理单个低延迟请求。因为批处理一定让推理更快，是否应该增大 Pipeline 的 batch_size？

**Reference answer**

No. Pipeline batch inference may help, especially on a GPU, but faster performance is not guaranteed. The guide advises against batching for latency-constrained live products and CPU use, and says to measure on the actual model, data and hardware.

**参考答案（中文）**

不应该这样假定。Pipeline 批处理可能有帮助，尤其是在 GPU 上，但不保证更快。文档不建议对延迟敏感的实时产品和 CPU 场景使用批处理，并要求在实际模型、数据和硬件上测量。

**Distractor**

A throughput-oriented setting is treated as a universal latency improvement.

**干扰点（中文）**

把面向吞吐量的设置当作普遍有效的延迟优化。

**Source:** [pipeline_tutorial.md, lines 171–171](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L171)

**来源：** 同上文档第 171–171 行；以下译文对应所列原文。

**Original evidence**

````text
[`Pipeline`] can also process batches of inputs with the `batch_size` parameter. Batch inference may improve speed, especially on a GPU, but it isn't guaranteed. Other variables such as hardware, data, and the model itself can affect whether batch inference improves speed. For this reason, batch inference is disabled by default.
````

**原文依据（中文译文）**

Pipeline 可以用 batch_size 参数批量处理输入。批量推理可能提速，尤其是在 GPU 上，但并不保证如此。硬件、数据和模型都会影响效果，因此默认关闭批量推理。

**Source:** [pipeline_tutorial.md, lines 206–211](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L206)

**来源：** 同上文档第 206–211 行；以下译文对应所列原文。

**Original evidence**

````text
1. The only way to know for sure is to measure performance on your model, data, and hardware.
2. Don't batch inference if you're constrained by latency (a live inference product for example).
3. Don't batch inference if you're using a CPU.
4. Don't batch inference if you don't know the `sequence_length` of your data. Measure performance, iteratively add to `sequence_length`, and include out-of-memory (OOM) checks to recover from failures.
5. Do batch inference if your `sequence_length` is regular, and keep pushing it until you reach an OOM error. The larger the GPU, the more helpful batch inference is.
6. Do make sure you can handle OOM errors if you decide to do batch inference.
````

**原文依据（中文译文）**

建议：唯一可靠的判断方法是在自己的模型、数据和硬件上测量。如果受延迟约束（例如实时推理产品），不要批量推理；使用 CPU 时也不要批量推理。如果不知道数据的 sequence_length，应先测量、逐步增加长度，并处理内存不足（OOM）以恢复。序列长度规律时可以尝试增大批量直至 OOM；GPU 越大，批量通常越有帮助。决定批量处理时，务必能处理 OOM 错误。

## distractor_020

**Focus**

Distinguish strict single-graph compilation from permissive graph splitting.

**考查点（中文）**

区分严格的单图编译和允许拆图的编译。

**Question**

With fullgraph=True, will torch.compile quietly split the model into several graphs when it encounters a graph break?

**问题（中文）**

设置 fullgraph=True 后，torch.compile 遇到计算图中断时，会悄悄把模型拆成多个图吗？

**Reference answer**

No. fullgraph=True attempts to compile the entire model into one graph. torch.compile raises an error if it encounters a graph break instead of quietly accepting several graphs.

**参考答案（中文）**

不会。fullgraph=True 要求尝试把整个模型编译成单个图。遇到计算图中断时，torch.compile 会报错，而不是默默接受多个图。

**Distractor**

The ordinary idea of graph splitting is incorrectly carried into strict fullgraph mode.

**干扰点（中文）**

把一般的拆图行为错误地套用到严格的 fullgraph 模式。

**Source:** [perf_torch_compile.md, lines 51–51](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_torch_compile.md#L51)

**来源：** 同上文档第 51–51 行；以下译文对应所列原文。

**Original evidence**

````text
Fullgraph attempts to compile the entire model into a single graph to maximize performance. torch.compile raises an error if it encounters a graph break, which means it can't compile the model into a single graph.
````

**原文依据（中文译文）**

Fullgraph 尝试把整个模型编译成单个图，以最大化性能。如果遇到计算图中断，说明无法编译成单图，torch.compile 会报错。

