# Golden dataset — 60 source-grounded draft questions

English questions and reference answers for the fixed local Transformers v4.57.1 corpus. Semantic: 20; keyword: 20; distractor: 20. All 30 selected documents are covered. These are authored references, not measured model results; independent human review remains pending.

Semantic tests intent-to-document matching without the target API name in the question. Keyword tests exact identifiers and values. Distractor tests similar features, false premises, conditions and incompatible combinations. Categories describe authoring intent and do not guarantee which retrieval mode will win. All questions are answerable from this corpus.

The runtime JSON contains only id, category, question and reference_answer. The companion sources JSON and this review document are outside data/documents and must not be indexed. Each source record includes a content hash and exact local excerpt for checking.

## semantic_001

**Focus:** Map an offline deployment requirement to preparation and loading controls.

**Question:** Our deployment environment cannot reach the internet. How can we prepare a pretrained model beforehand and ensure loading does not contact the hosting service?

**Reference answer:** Download the model repository and cache its files before going offline, for example with snapshot_download. Set HF_HUB_OFFLINE=1 to prevent HTTP calls to the Hub, or use from_pretrained(..., local_files_only=True) to load only locally cached files. Offline operation requires the needed files to already be available.

**Source:** [installation.md — Offline mode](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/installation.md#L142)

```text
To use Transformers in an offline or firewalled environment requires the downloaded and cached files ahead of time. Download a model repository from the Hub with the [`~huggingface_hub.snapshot_download`] method.
```

**Source:** [installation.md — Offline mode](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/installation.md#L153)

```text
Set the environment variable `HF_HUB_OFFLINE=1` to prevent HTTP calls to the Hub when loading a model.
```

**Source:** [installation.md — Offline mode](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/installation.md#L160)

```text
Another option for only loading cached files is to set `local_files_only=True` in [`~PreTrainedModel.from_pretrained`].
```

## semantic_002

**Focus:** Understand automatic placement from a memory-limited deployment scenario.

**Question:** A pretrained model cannot fit entirely on our graphics card. How can loading spread its weights across available hardware instead of requiring everything on that card?

**Reference answer:** Load the model with from_pretrained(..., device_map="auto") to enable Accelerate Big Model Inference. Weights are dispatched to the fastest devices first, normally GPUs, and remaining weights can be offloaded to CPU and disk. This also avoids retaining separate random and pretrained copies during loading.

**Source:** [models.md — Big Model Inference](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L200)

```text
Big Model Inference's second feature relates to how weights are loaded and dispatched in the model skeleton. Model weights are dispatched across all available devices, starting with the fastest device (usually the GPU) and then offloading any remaining weights to slower devices (CPU and hard drive).
```

**Source:** [models.md — Big Model Inference](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L204)

```text
Set [device_map](https://github.com/huggingface/transformers/blob/026a173a64372e9602a16523b8fae9de4b0ff428/src/transformers/modeling_utils.py#L3061) to `"auto"` to enable Big Model Inference.
```

**Source:** [models.md — Big Model Inference](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L195)

```text
Randomly initialized weights are only created when the pretrained weights are loaded to avoid maintaining two copies of the model in memory at the same time. The maximum memory usage is only the size of the model.
```

## semantic_003

**Focus:** Explain the purpose of subword representation from an unseen-word scenario.

**Question:** Why can a text model handle an unfamiliar long word without keeping every possible word in its vocabulary?

**Reference answer:** Subword tokenization represents rare or unseen words as combinations of known subwords. This keeps vocabulary size manageable while retaining meaningful representations, rather than requiring every whole word to be present or treating text only as individual characters.

**Source:** [tokenizer_summary.md — Subword tokenization](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/tokenizer_summary.md#L104)

```text
Subword tokenization allows the model to have a reasonable vocabulary size while being able to learn meaningful
context-independent representations. In addition, subword tokenization enables the model to process words it has never
seen before, by decomposing them into known subwords. For instance, the [`~transformers.BertTokenizer`] tokenizes
`"I have a new GPU!"` as follows:
```

## semantic_004

**Focus:** Connect user-visible waiting to progressive output.

**Question:** In a user-facing application, people currently see nothing until an entire answer finishes. What documented generation feature can make output appear progressively?

**Reference answer:** Use streaming. Construct a TextStreamer with the tokenizer and pass it as the streamer argument to model.generate. It returns text as generation progresses rather than waiting for the complete response, reducing perceived latency.

**Source:** [generation_features.md — Streaming](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L25)

```text
Streaming starts returning text as soon as it is generated so you don't have to wait to see the entire generated response all at once. It is important in user-facing applications because it reduces perceived latency and allows users to see the generation progression.
```

**Source:** [generation_features.md — Streaming](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L34)

```text
Create an instance of [`TextStreamer`] with the tokenizer. Pass [`TextStreamer`] to the `streamer` parameter in [`~GenerationMixin.generate`] to stream the output one word at a time.
```

## semantic_005

**Focus:** Diagnose a conversation boundary problem without naming its control in the question.

**Question:** A conversational model sometimes continues the user’s sentence instead of replying. How can we make the formatted conversation indicate that the next text should be an assistant response?

**Reference answer:** Use the model’s chat template with add_generation_prompt=True to append tokens marking the start of an assistant response. This tells the model to reply instead of continuing the user message. Some models do not need such a prefix, so the option has no effect for those templates.

**Source:** [chat_templating.md — add_generation_prompt](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L131)

```text
You may have noticed the [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) argument in the above examples.
This argument adds tokens to the end of the chat that indicate the start of an `assistant` response. Remember: Beneath all the chat abstractions, chat models are still just language models that continue a sequence of tokens!
If you include tokens that tell it that it's now in an `assistant` response, it will correctly write a response, but if you don't include these tokens, the model may get confused and do something strange, like **continuing** the user's message instead of replying to it!
```

**Source:** [chat_templating.md — add_generation_prompt](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L171)

```text
Not all models require generation prompts, and some models, like [Llama](./model_doc/llama), don't have any special tokens before the `assistant` response. In these cases, [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) has no effect.
```

## semantic_006

**Focus:** Relate limited training resources and small deliverables to adapters.

**Question:** We want to adapt a large pretrained model on a consumer GPU and distribute only a small task-specific addition. What approach does the documentation recommend, and why does it save memory?

**Reference answer:** Use parameter-efficient fine-tuning with adapters, such as LoRA. Only a small number of extra parameters are trained on top of the pretrained base model. The GPU does not need optimizer states and gradients for the base model, and the lightweight adapters are convenient to store, share and load.

**Source:** [peft.md — PEFT](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/peft.md#L16)

```text
[PEFT](https://huggingface.co/docs/peft/index), a library of parameter-efficient fine-tuning methods, enables training and storing large models on consumer GPUs. These methods only fine-tune a small number of extra model parameters, also known as adapters, on top of the pretrained model. A significant amount of memory is saved because the GPU doesn't need to store the optimizer states and gradients for the pretrained base model. Adapters are very lightweight, making it convenient to share, store, and load them.
```

## semantic_007

**Focus:** Distinguish an effective training batch from a simultaneously resident batch.

**Question:** Our training examples fit only in small batches, but we want updates equivalent to using a larger batch. How can we achieve that without placing the whole larger batch in memory at once?

**Reference answer:** Use gradient accumulation: accumulate gradients across multiple mini-batches before updating model parameters. This produces a larger effective batch size while processing smaller batches. It can slow training because multiple forward and backward passes are needed for each update.

**Source:** [perf_train_gpu_one.md — Gradient accumulation](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_train_gpu_one.md#L63)

```text
Gradient accumulation overcomes memory constraints - useful for fitting a very large model that otherwise wouldn't fit on a single GPU - by accumulating gradients over multiple mini-batches before updating the parameters. This reduces memory by storing fewer gradients and enables training with a larger *effective batch size* because usually, the parameters are updated from a single batch of data. Training can slow down though due to the additional forward and backward passes introduced by gradient accumulation.
```

## semantic_008

**Focus:** Identify recomputation as the trade-off for activation storage.

**Question:** Intermediate activations are consuming too much memory during training. What technique can reduce their storage, and what cost should we expect?

**Reference answer:** Enable gradient checkpointing. Store only some intermediate activations and recompute the remaining ones when needed during the backward pass. This reduces activation memory at the cost of slower training; the guide describes an approximately 20 percent slowdown.

**Source:** [perf_train_gpu_one.md — Gradient checkpointing](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_train_gpu_one.md#L86)

```text
Gradient checkpointing reduces memory usage by only storing some of the intermediate activations during the backward pass and recomputing the remaining activations. This avoids storing *all* of the intermediate activations from the forward pass, which can require a lot of memory overhead. However, it comes at the cost of slower training speed (~20%).
```

## semantic_009

**Focus:** Interpret cold-start latency rather than confusing it with steady-state inference.

**Question:** An optimized model is slow on its first call but much faster on subsequent calls. What documented initialization cost can explain this pattern?

**Reference answer:** torch.compile needs to compile the model initially, so the initial call is slow. Later calls can reuse the compiled model rather than compiling again, making them faster. The one-time compilation cost should be distinguished from subsequent execution performance.

**Source:** [perf_torch_compile.md — torch.compile](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_torch_compile.md#L29)

```text
> [!TIP]
> The initial call to torch.compile is slow because the model needs to be compiled. Subsequent calls to the compiled model are much faster because it doesn't need to compile again.
```

## semantic_010

**Focus:** Explain weight precision reduction without requiring an algorithm name in the question.

**Question:** A model’s weight storage is too large for deployment. What general compression approach can lower its memory requirements, and what property is it trying to retain?

**Reference answer:** Quantization stores model weights in lower precision to reduce memory requirements while trying to retain as much accuracy as possible. Depending on the method, weights may use lower-precision floating point or integer representations such as int8 or int4. Methods differ in calibration requirements and other trade-offs.

**Source:** [quantization/overview.md — Overview](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/overview.md#L19)

```text
Quantization lowers the memory requirements of loading and using a model by storing the weights in a lower precision while trying to preserve as much accuracy as possible. Weights are typically stored in full-precision (fp32) floating point representations, but half-precision (fp16 or bf16) are increasingly popular data types given the large size of models today. Some quantization methods can reduce the precision even further to integer representations, like int8 or int4.
```

**Source:** [quantization/overview.md — Overview](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/overview.md#L21)

```text
Transformers supports many quantization methods, each with their pros and cons, so you can pick the best one for your specific use case. Some methods require calibration for greater accuracy and extreme compression (1-2 bits), while other methods work out of the box with on-the-fly quantization.
```

## semantic_011

**Focus:** Map restoration of compressed weights to the operation and its caveats.

**Question:** We want to restore a compressed model to its original precision. What operation is documented, and what must we consider before using it?

**Reference answer:** Call the model’s dequantize method to restore the original precision. Ensure enough GPU memory is available to hold the dequantized model, and note that dequantization may result in some quality loss.

**Source:** [quantization/bitsandbytes.md — Dequantizing bitsandbytes models](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L329)

```text
Once quantized, you can [`~PreTrainedModel.dequantize`] a model to the original precision but this may result in some quality loss. Make sure you have enough GPU memory to fit the dequantized model.
```

## semantic_012

**Focus:** Map cross-framework deployment to the export and execution tools.

**Question:** We need to deploy a model with a runtime that is separate from the original training framework. What export format and supporting library are described for this purpose?

**Reference answer:** Export the model to ONNX, an open standard for representing models through common operators and a computational graph across frameworks. The Optimum library provides export support, and ONNX Runtime can run the exported model through task-specific ORTModel classes.

**Source:** [serialization.md — ONNX](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/serialization.md#L19)

```text
[ONNX](http://onnx.ai) is an open standard that defines a common set of operators and a file format to represent deep learning models in different frameworks, including PyTorch and TensorFlow. When a model is exported to ONNX, the operators construct a computational graph (or *intermediate representation*) which represents the flow of data through the model. Standardized operators and data types makes it easy to switch between frameworks.
```

**Source:** [serialization.md — ONNX](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/serialization.md#L21)

```text
The [Optimum](https://huggingface.co/docs/optimum/index) library exports a model to ONNX with configuration objects which are supported for [many architectures](https://huggingface.co/docs/optimum/exporters/onnx/overview) and can be easily extended. If a model isn't supported, feel free to make a [contribution](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute) to Optimum.
```

**Source:** [serialization.md — ONNX](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/serialization.md#L25)

```text
- [Graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) and [quantization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization) for improving inference.
- Use the [`~optimum.onnxruntime.ORTModel`] API to run a model with [ONNX Runtime](https://onnxruntime.ai/).
- Use [optimized inference pipelines](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines) for ONNX models.
```

## semantic_013

**Focus:** Relate preprocessing throughput to batching and implementation.

**Question:** Our preprocessing is a bottleneck because we encode text one sentence at a time. How should we organize it, and which tokenizer implementation is especially useful?

**Reference answer:** Preprocess batches of text rather than one sentence at a time. Fast tokenizers are Rust-based implementations and are especially effective at parallelizing batched tokenization; they also provide additional alignment methods compared with Python tokenizers.

**Source:** [fast_tokenizers.md — Batch tokenization](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L311)

```text
It is faster and more efficient to preprocess *batches* of text instead of a single sentence at a time. Fast tokenizers are especially good at parallelizing tokenization.
```

**Source:** [fast_tokenizers.md — Fast tokenizers](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L136)

```text
[`PreTrainedTokenizerFast`] or *fast tokenizers* are Rust-based tokenizers from the [Tokenizers](https://hf.co/docs/tokenizers) library. It is significantly faster at batched tokenization and provides additional alignment methods compared to the Python-based tokenizers.
```

## semantic_014

**Focus:** Recover a stopped training workflow.

**Question:** A training job stopped unexpectedly, but its saved checkpoints are still available. How can we continue from saved progress rather than restarting from scratch?

**Reference answer:** Call trainer.train with resume_from_checkpoint=True to resume from the last checkpoint, or pass a specific checkpoint path to resume from that checkpoint. Saved checkpoints are intended to support recovery and continuation of training.

**Source:** [trainer.md — Checkpoints](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/trainer.md#L87)

```text
Saving checkpoints are useful for resuming training or recovering your training progress if you encounter an error. Set the `resume_from_checkpoint` parameter in [`~Trainer.train`] to resume training from the last checkpoint or a specific checkpoint.
```

**Source:** [trainer.md — Checkpoints](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/trainer.md#L92)

```text
```py
trainer.train(resume_from_checkpoint=True)
```
```

## semantic_015

**Focus:** Interpret task-head initialization as a normal adaptation step.

**Question:** We loaded a general pretrained encoder for review classification and received a warning that the new classification layer has uninitialized weights. Why is this expected, and what should we do?

**Reference answer:** The pretrained head is replaced by a randomly initialized classification head for the new task. That head must be fine-tuned on the classification dataset before it can produce meaningful predictions. Load the sequence-classification model with the appropriate number of labels and train it.

**Source:** [training.md — Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L77)

```text
> [!TIP]
> The message above is a reminder that the models pretrained head is discarded and replaced with a randomly initialized classification head. The randomly initialized head needs to be fine-tuned on your specific task to output meaningful predictions.
```

**Source:** [training.md — Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L72)

```text
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
"Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']"
"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
```
```

## semantic_016

**Focus:** Interpret benchmark results in the context of the actual workload.

**Question:** We found a published comparison in which one compression method is faster. How should we decide whether that method meets our own deployment requirements?

**Reference answer:** Benchmark the quantized model’s accuracy and speed on your specific task and hardware instead of treating the published comparison as a universal guarantee. Check whether the measured results meet your own requirements and consult the method’s usage documentation.

**Source:** [quantization/selecting.md — Benchmark Comparison](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L156)

```text
> [!TIP]
> Always benchmark the performance (accuracy and speed) of the quantized model on your specific task and hardware to ensure it meets your requirements. Refer to the individual documentation pages linked above for detailed usage instructions.
```

## semantic_017

**Focus:** Diagnose long-context state memory rather than weight memory.

**Question:** Long conversations are exhausting GPU memory because their accumulated attention state is large. How can we keep less of that state on the GPU, and what performance trade-off follows?

**Reference answer:** Use cache offloading. Most layers’ KV caches are moved to CPU, while the current layer’s cache is maintained on GPU and the next layer is prefetched. This saves GPU memory, but transfers between CPU and GPU can reduce generation throughput.

**Source:** [kv_cache.md — Cache offloading](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L95)

```text
Offloading the cache saves GPU memory by moving the KV cache for model layers except one to the CPU. Only the current layer cache is maintained on the GPU during a models `forward` iteration over the layers. It will asynchronously prefetch the next layer's cache, and send back the current layer's cache back to the CPU after attention computation.
```

**Source:** [kv_cache.md — Cache offloading](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L99)

```text
> [!WARNING]
> You may notice a small degradation in generation throughput compared to a full on-device cache, depending on your model and generation choices (context size, number of generated tokens, number of beams, etc.). This is because moving the key/value states back and forth requires some work.
```

## semantic_018

**Focus:** Assess workload shape when choosing fixed cache capacity.

**Question:** Our generation requests have very different lengths. Why might reserving one large fixed attention-state buffer be inefficient even though it helps compilation?

**Reference answer:** A StaticCache preallocates a maximum size and supports compiling the decoding stage. However, unused positions are masked and still cause wasted attention computation, so a large fixed cache can be inefficient for many short requests mixed with a long request. It is a trade-off that is better suited to similar-length sequences.

**Source:** [kv_cache.md — Fixed-size cache](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L72)

```text
A fixed-size cache ([`StaticCache`]) pre-allocates a specific maximum cache size for the kv pairs. You can generate up to the maximum cache size without needing to modify it. However, having a fixed (usually large) size for the key/value states means that while generating, a lot of tokens will actually be masked as they should not take part in the attention. So this trick allows to easily `compile` the decoding stage, but it incurs a waste of tokens in the attention computation. As all things, it's then a trade-off which should be very good if you generate with several sequence of more or less the same lengths, but may be sub-optimal if you have for example 1 very large sequence, and then only short sequences (as the fix cache size would be large, a lot would be wasted for the short sequences). Make sure you understand the impact if you use it!
```

## semantic_019

**Focus:** Connect external optimizer installation to the native training interface.

**Question:** We want to use an optimizer supplied by an external library with the standard training loop. What integration steps does the guide describe?

**Reference answer:** Install the library that supplies the optimizer and choose its identifier with the optim parameter in TrainingArguments. Depending on the optimizer, configure additional settings such as optim_target_modules or optim_args as documented, then use those training arguments with Trainer.

**Source:** [optimizers.md — Optimizers](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/optimizers.md#L19)

```text
Transformers offers two native optimizers, AdamW and AdaFactor. It also provides integrations for more specialized optimizers. Install the library that offers the optimizer and drop it in the `optim` parameter in [`TrainingArguments`].
```

**Source:** [optimizers.md — APOLLO](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/optimizers.md#L50)

```text
Use the `optim_target_modules` parameter to specify which layers to train.
```

**Source:** [optimizers.md — APOLLO](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/optimizers.md#L70)

```text
For additional training options, use `optim_args` to define hyperparameters like `rank`, `scale`, and more. Refer to the table below for a complete list of available hyperparameters.
```

## semantic_020

**Focus:** Understand the coupled interfaces behind a custom attention backend.

**Question:** A custom attention implementation needs a different attention-mask format. What additional registration is needed so the model can construct the appropriate mask?

**Reference answer:** Register a corresponding mask function with AttentionMaskInterface in addition to the custom attention function. The mask interface adapts mask creation to the selected attention implementation. If no matching mask function is registered, mask creation is skipped.

**Source:** [attention_interface.md — Attention Mask Interface](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L160)

```text
Having a new attention function may mean that you need a new format of attention mask to decide what key and value tokens
the query tokens should attend to. This is now possible with the `AttentionMaskInterface`! It works in the same way as
the `AttentionInterface`:
```

**Source:** [attention_interface.md — Attention Mask Interface](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L176)

```text
The reason you have to register it is because we need to automatically correct your mask format based on the attention implementation (for example, flex attention uses a BlockMask format, while sdpa uses a 4D tensor).
By default, if you do not register an attention mask function along with your attention function, mask creation will be skipped
and `attention_mask=None` will be passed along to the Attention layers.
```

## keyword_001

**Focus:** Exact padding enum behavior.

**Question:** What do padding="longest" and padding="max_length" do when the tokenizer receives a single sequence?

**Reference answer:** padding="longest" (or True) applies no padding to a single sequence. padding="max_length" still pads a single sequence to the specified max_length, or to the model’s maximum length if max_length is not provided.

**Source:** [pad_truncation.md — Padding and truncation](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L25)

```text
- `True` or `'longest'`: pad to the longest sequence in the batch (no padding is applied if you only provide
    a single sequence).
- `'max_length'`: pad to a length specified by the `max_length` argument or the maximum length accepted
    by the model if no `max_length` is provided (`max_length=None`). Padding will still be applied if you only provide a single sequence.
- `False` or `'do_not_pad'`: no padding is applied. This is the default behavior.
```

## keyword_002

**Focus:** Precise truncation enum.

**Question:** For a pair of sequences, which truncation value preserves the first sequence and truncates only the second?

**Reference answer:** Use truncation="only_second". It truncates only the second sequence of a pair to satisfy the specified max_length, or the model’s maximum accepted length when max_length is not provided.

**Source:** [pad_truncation.md — Padding and truncation](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L33)

```text
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
```

## keyword_003

**Focus:** Locate a task-specific return parameter.

**Question:** Which Pipeline parameter value returns only newly generated text instead of including the prompt?

**Reference answer:** Pass return_full_text=False to the text-generation pipeline. It returns the generated continuation rather than the prompt plus generated text.

**Source:** [pipeline_tutorial.md — Task-specific parameters](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L256)

```text
Pass `return_full_text=False` to [`Pipeline`] to only return the generated text instead of the full text (prompt and generated text).
```

## keyword_004

**Focus:** Exact ASR parameter value.

**Question:** What return_timestamps value requests word-level timestamps from an automatic speech recognition pipeline?

**Reference answer:** Use return_timestamps="word". It requests timestamps indicating when each word was spoken.

**Source:** [pipeline_tutorial.md — Task-specific parameters](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L222)

```text
Pass the `return_timestamps="word"` parameter to [`Pipeline`] to return when each word was spoken.
```

## keyword_005

**Focus:** Exact loading parameter semantics.

**Question:** What is the documented purpose of dtype="auto" when loading pretrained model weights?

**Reference answer:** dtype="auto" initializes weights directly in the data type in which they are stored. This can avoid loading them twice, first in PyTorch’s default float32 and then in the desired data type.

**Source:** [quicktour.md — Pretrained models](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quicktour.md#L86)

```text
- `device_map="auto"` automatically allocates the model weights to your fastest device first.
- `dtype="auto"` directly initializes the model weights in the data type they're stored in, which can help avoid loading the weights twice (PyTorch loads weights in `torch.float32` by default).
```

## keyword_006

**Focus:** Exact checkpoint parameter and value.

**Question:** What is the documented default max_shard_size in save_pretrained, and what does that parameter control?

**Reference answer:** max_shard_size defaults to 5GB for each checkpoint shard. It controls the shard size used when saving model weights, helping the resulting checkpoint fit memory-limited environments.

**Source:** [models.md — Sharded checkpoints](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L119)

```text
The `max_shard_size` parameter defaults to 5GB for each shard because it is easier to run on free-tier GPU instances without running out of memory.
```

## keyword_007

**Focus:** Framework tensor enum.

**Question:** Which tokenizer return_tensors value requests PyTorch tensors?

**Reference answer:** Set return_tensors="pt" when calling the tokenizer to request PyTorch tensors.

**Source:** [fast_tokenizers.md — Tokenizers](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L31)

```text
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt")
{'input_ids': tensor([[     2,   1734,    708,   1508,   4915,    577,   1500,    692,    573,
         156808, 128149,   9581, 235265]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```
```

## keyword_008

**Focus:** Exact flag for two-step chat formatting.

**Question:** After apply_chat_template(tokenize=False), what add_special_tokens value should be used if the formatted string is tokenized later?

**Reference answer:** Set add_special_tokens=False. The chat template should already contain the required special tokens, so adding them again can duplicate tokens and hurt model performance.

**Source:** [chat_templating.md — Using `apply_chat_template`](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L125)

```text
> [!WARNING]
> Some tokenizers add special `<bos>` and `<eos>` tokens. Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. When you format text with `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` if you tokenize later to avoid duplicating these tokens.
> This isn't an issue if you use `apply_chat_template(tokenize=True)`, which means it's usually the safer option!
```

## keyword_009

**Focus:** Exact Jinja helper names and outputs.

**Question:** What do raise_exception(msg) and strftime_now(format_str) do inside a chat template?

**Reference answer:** raise_exception(msg) raises a TemplateException, useful for debugging or reporting incorrect template usage. strftime_now(format_str) retrieves the current date and time formatted with the supplied format string.

**Source:** [chat_templating_writing.md — Special variables and callables](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating_writing.md#L119)

```text
- `raise_exception(msg)` raises a `TemplateException`. This is useful for debugging or warning users about incorrect template usage.
- `strftime_now(format_str)` retrieves the current date and time in a specific format, which is often required in system messages. It is equivalent to [datetime.now().strftime(format_str)](https://docs.python.org/3/library/datetime.html#datetime.datetime.now) in Python.
```

## keyword_010

**Focus:** Identify the generation-length parameter.

**Question:** What does max_new_tokens control in text generation?

**Reference answer:** max_new_tokens controls the maximum number of newly generated tokens. Set it explicitly when a longer response is needed instead of relying on a potentially small generation default.

**Source:** [llm_tutorial.md — Common Options](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_tutorial.md#L150)

```text
| Option name | Type | Simplified description |
|---|---|---|
| `max_new_tokens` | `int` | Controls the maximum generation length. Be sure to define it, as it usually defaults to a small value. |
| `do_sample` | `bool` | Defines whether generation will sample the next token (`True`), or is greedy instead (`False`). Most use cases should set this flag to `True`. Check [this guide](./generation_strategies) for more information. |
| `temperature` | `float` | How unpredictable the next selected token will be. High values (`>0.8`) are good for creative tasks, low values (e.g. `<0.4`) for tasks that require "thinking". Requires `do_sample=True`. |
| `num_beams` | `int` | When set to `>1`, activates the beam search algorithm. Beam search is good on input-grounded tasks. Check [this guide](./generation_strategies) for more information. |
| `repetition_penalty` | `float` | Set it to `>1.0` if you're seeing the model repeat itself often. Larger values apply a larger penalty. |
| `eos_token_id` | `list[int]` | The token(s) that will cause generation to stop. The default value is usually good, but you can specify a different token. |
```

## keyword_011

**Focus:** Exact decoding configuration.

**Question:** What settings enable multinomial sampling rather than beam search in generate?

**Reference answer:** Set do_sample=True and num_beams=1. This samples tokens from the model’s probability distribution without maintaining multiple beams.

**Source:** [generation_strategies.md — Sampling](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_strategies.md#L53)

```text
Enable multinomial sampling with `do_sample=True` and `num_beams=1`.
```

## keyword_012

**Focus:** Similar cache enum identifiers.

**Question:** Which cache_implementation values enable offloading for dynamic and static caches respectively?

**Reference answer:** Use cache_implementation="offloaded" for the dynamic cache and cache_implementation="offloaded_static" for the static cache. These can be configured in GenerationConfig or generate.

**Source:** [kv_cache.md — Cache offloading](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L102)

```text
Offloading is available for both [`DynamicCache`] and [`StaticCache`]. You can enable it by configuring `cache_implementation="offloaded"` for the dynamic version, or `cache_implementation="offloaded_static"` for the static version, in either [`GenerationConfig`] or [`~GenerationMixin.generate`].
Additionally, you can also instantiate your own [`DynamicCache`] or [`StaticCache`] with the `offloading=True` option, and pass this cache in `generate` or your model's `forward` (for example, `past_key_values=DynamicCache(config=model.config, offloading=True)` for a dynamic cache).
```

## keyword_013

**Focus:** Exact shape and dimension meaning.

**Question:** What attention-mask shape is required when implementing an iterative forward loop with a KV cache?

**Reference answer:** The attention mask must have shape (batch_size, past_kv_length + new_tokens_length), covering both previously cached tokens and current tokens.

**Source:** [cache_explanation.md — Cache class](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/cache_explanation.md#L78)

```text
2. When the `forward` method is called iteratively, it's crucial that the attention mask shape matches the combined length of the past and current kv pairs. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is typically handled internally in [`~GenerationMixin.generate`], but if you want to implement your own generation loop with [`Cache`], keep this in mind! The attention mask should hold the past and current token values.
```

## keyword_014

**Focus:** Precise training-evaluation API name.

**Question:** Which TrainingArguments setting in the classification example reports evaluation accuracy at the end of each epoch?

**Reference answer:** Use eval_strategy="epoch" in TrainingArguments. Supply a compute_metrics function to Trainer to compute the classification metric.

**Source:** [training.md — Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L99)

```text
Set up [`TrainingArguments`] with where to save the model and when to compute accuracy during training. The example below sets it to `"epoch"`, which reports the accuracy at the end of each epoch. Add `push_to_hub=True` to upload the model to the Hub after training.
```

**Source:** [training.md — Trainer](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/training.md#L113)

```text
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
```

## keyword_015

**Focus:** Exact stage-number behavior.

**Question:** What does ZeRO stage 1 shard, and what extra states does ZeRO stage 2 shard?

**Reference answer:** ZeRO-1 shards optimizer states across GPUs. ZeRO-2 shards both optimizer states and gradient states across GPUs.

**Source:** [deepspeed.md — ZeRO stage](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/deepspeed.md#L166)

```text
ZeRO-1 shards the optimizer states across GPUs and you can expect a small speed up.
```

**Source:** [deepspeed.md — ZeRO stage](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/deepspeed.md#L179)

```text
ZeRO-2 shards the optimizer and gradient states across GPUs. This stage is primarily used for training since its features are not relevant to inference. Some important parameters to configure for better performance include the following.
```

## keyword_016

**Focus:** Exact runtime switching method.

**Question:** How can set_attn_implementation switch an already loaded model back to SDPA?

**Reference answer:** Call model.set_attn_implementation("sdpa"). This changes the attention function without needing to reload the model.

**Source:** [attention_interface.md — Dynamically switching attention function](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L65)

```text
```python
# Back to use original sdpa implementation
model.set_attn_implementation("sdpa")
```

**Source:** [attention_interface.md — Dynamically switching attention function](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/attention_interface.md#L72)

```text
and it will stop printing the statements, as it now uses the `sdpa` attention.  
This allows to quickly change an attention function, without needing to reload the model!
```

## keyword_017

**Focus:** Exact nested-quantization flag and documented amount.

**Question:** Which BitsAndBytesConfig option enables nested four-bit quantization, and how much extra storage does the guide say it saves?

**Reference answer:** Set bnb_4bit_use_double_quant=True alongside load_in_4bit=True. Nested quantization quantizes already quantized weights again and saves an additional 0.4 bits per parameter.

**Source:** [quantization/bitsandbytes.md — Nested quantization](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L319)

```text
double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)
```

**Source:** [quantization/bitsandbytes.md — Nested quantization](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L314)

```text
Nested quantization can save additional memory at no additional performance cost. This feature performs a second quantization of the already quantized weights to save an additional 0.4 bits/parameter. For example, with nested quantization, you can finetune a [Llama-13b](https://huggingface.co/meta-llama/Llama-2-13b) model on a 16GB NVIDIA T4 GPU with a sequence length of 1024, a batch size of 1, and enable gradient accumulation with 4 steps.
```

## keyword_018

**Focus:** Exact backend enum and bit-width restriction.

**Question:** What value of GPTQConfig.backend activates the Marlin kernel, and what bit width does that kernel support?

**Reference answer:** Set backend="marlin" in GPTQConfig. The documented Marlin kernel supports four-bit GPTQ inference and is optimized for NVIDIA A100 GPUs.

**Source:** [quantization/gptq.md — Marlin](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/gptq.md#L121)

```text
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=GPTQConfig(bits=4, backend="marlin"))
```
```

**Source:** [quantization/gptq.md — Marlin](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/gptq.md#L113)

```text
[Marlin](https://github.com/IST-DASLab/marlin) is a 4-bit only CUDA GPTQ kernel, highly optimized for the NVIDIA A100 GPU (Ampere) architecture. Loading, dequantization, and execution of post-dequantized weights are highly parallelized, offering a substantial inference improvement versus the original CUDA GPTQ kernel. Marlin is only available for quantized inference and does not support model quantization.
```

## keyword_019

**Focus:** Exact AWQ fusion controls.

**Question:** What do AwqConfig.do_fuse and fuse_max_seq_len configure?

**Reference answer:** do_fuse=True enables fused AWQ modules. fuse_max_seq_len is the total sequence length, including both the input context and expected generated text; choose a sufficiently large value.

**Source:** [quantization/awq.md — Fused modules](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/awq.md#L92)

```text
Create an [`AwqConfig`] and set the parameters `fuse_max_seq_len` and `do_fuse=True` to enable fused modules. The `fuse_max_seq_len` parameter is the total sequence length and it should include the context length and the expected generation length. Set it to a larger value to be safe.
```

## keyword_020

**Focus:** Locate an optimization-specific tokenizer parameter.

**Question:** What input-padding control is recommended with static KV cache and torch.compile to reduce shape-related recompilations?

**Reference answer:** Left-pad inputs using pad_to_multiple_of to a limited set of values. This reduces changes in input shapes that can trigger recompilation.

**Source:** [llm_optims.md — Static kv-cache and torch.compile](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_optims.md#L40)

```text
> [!TIP]
> Regardless of how you use the static kv-cache and torch.compile, left-pad your inputs with [pad_to_multiple_of](https://hf.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of) to a limited set of values to avoid shape-related recompilations.
```

## distractor_001

**Focus:** Reject conflating padding with truncation.

**Question:** A developer sets padding=True and claims this alone will shorten overlong inputs to the model limit. Is that correct, and what control is missing?

**Reference answer:** No. Padding adds padding tokens to shorter sequences; it does not shorten long ones. Enable truncation=True, optionally with max_length, to truncate overlong inputs. Padding and truncation solve opposite length problems.

**Distractor:** Padding configuration is plausible but does not truncate.

**Source:** [pad_truncation.md — Padding and truncation](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L19)

```text
Batched inputs are often different lengths, so they can't be converted to fixed-size tensors. Padding and truncation are strategies for dealing with this problem, to create rectangular tensors from batches of varying lengths. Padding adds a special **padding token** to ensure shorter sequences will have the same length as either the longest sequence in a batch or the maximum length accepted by the model. Truncation works in the other direction by truncating long sequences.
```

**Source:** [pad_truncation.md — Padding and truncation](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pad_truncation.md#L31)

```text
The `truncation` argument controls truncation. It can be a boolean or a string:
```

## distractor_002

**Focus:** Choose continuation rather than a new-message prefix.

**Question:** We want to continue an assistant message that already starts a JSON object. Should we enable add_generation_prompt and continue_final_message together to be safe?

**Reference answer:** No. Use continue_final_message=True to continue the existing final message. add_generation_prompt starts a new assistant message, whereas continue_final_message removes end-of-sequence markers to continue one. Combining the two returns an error.

**Distractor:** Both flags sound relevant, but their message-boundary operations conflict.

**Source:** [chat_templating.md — continue_final_message](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L177)

```text
This is useful for “prefilling” a model response. In the example below, the model generates text that continues the JSON string rather than starting a new message. It can be very useful for improving the accuracy of instruction following when you know how to start its replies.
```

**Source:** [chat_templating.md — continue_final_message](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L189)

```text
> [!WARNING]
> You shouldn't use [add_generation_prompt](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.add_generation_prompt) and [continue_final_message](https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template.continue_final_message) together. The former adds tokens that start a new message, while the latter removes end of sequence tokens. Using them together returns an error.
```

## distractor_003

**Focus:** Distinguish training formatting from inference prompting.

**Question:** For supervised training on completed conversations, should we keep add_generation_prompt=True because it is helpful when generating assistant replies?

**Reference answer:** No. Apply the chat template while preprocessing training data, but set add_generation_prompt=False. The extra tokens that prompt a new assistant response are not helpful during training on the supplied conversations.

**Distractor:** An inference setting is incorrectly transferred to training.

**Source:** [chat_templating.md — Model training](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/chat_templating.md#L196)

```text
Training a model with a chat template is a good way to ensure the template matches the tokens the model was trained on. Apply the chat template as a preprocessing step to your dataset. Set `add_generation_prompt=False` because the additional tokens to prompt an assistant response aren't helpful during training.
```

## distractor_004

**Focus:** Distinguish local token selection from sequence-level candidates.

**Question:** Is beam search the same as greedily choosing the highest-probability next token at every step? Explain the difference.

**Reference answer:** No. Greedy search selects the most likely next token at each step. Beam search tracks several candidate sequences and selects one with the highest overall probability, so it can retain a sequence whose early tokens are not locally the most likely.

**Distractor:** Both pursue probability, but optimize over different candidate structures.

**Source:** [generation_strategies.md — Greedy search](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_strategies.md#L29)

```text
Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in [`GenerationConfig`], this strategy generates a maximum of 20 new tokens.
```

**Source:** [generation_strategies.md — Beam search](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_strategies.md#L73)

```text
Beam search keeps track of several generated sequences (beams) at each time step. After a certain number of steps, it selects the sequence with the highest *overall* probability. Unlike greedy search, this strategy can "look ahead" and pick a sequence with a higher probability overall even if the initial tokens have a lower probability. It is best suited for input-grounded tasks, like describing an image or speech recognition. You can also use `do_sample=True` with beam search to sample at each step, but beam search will still greedily prune out low probability sequences between steps.
```

## distractor_005

**Focus:** Respect matched generation/detection settings.

**Question:** Can WatermarkDetector use an arbitrary WatermarkingConfig to detect text that was generated with another configuration?

**Reference answer:** No. WatermarkDetector must use the same WatermarkingConfig as the one used during generation. It detects the proportion of green tokens under that watermarking scheme.

**Distractor:** A detector with the right class name but wrong configuration is not equivalent.

**Source:** [generation_features.md — Watermarking](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L75)

```text
Create an instance of [`WatermarkDetector`] and pass the model output to it to detect whether the text is machine-generated. The [`WatermarkDetector`] must have the same [`WatermarkingConfig`] used during generation.
```

**Source:** [generation_features.md — Watermarking](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/generation_features.md#L57)

```text
> [!TIP]
> The [`WatermarkDetector`] class detects the proportion of green tokens in generated text, which is why it is recommended to strip the prompt text, if it is much longer than the generated text. Padding can also have an effect on [`WatermarkDetector`].
```

## distractor_006

**Focus:** Distinguish architecture configuration from trained checkpoint weights.

**Question:** Does constructing a model from its configuration give us the pretrained checkpoint’s learned weights?

**Reference answer:** No. A configuration describes the architecture and its attributes; constructing the model this way yields random weights that need training. Load learned checkpoint weights with from_pretrained instead.

**Distractor:** Architecture compatibility is mistaken for pretrained state.

**Source:** [models.md — Models and configurations](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L36)

```text
All models have a `configuration.py` file with specific attributes like the number of hidden layers, vocabulary size, activation function, and more. You'll also find a `modeling.py` file that defines the layers and mathematical operations taking place inside each layer. The `modeling.py` file takes the model attributes in `configuration.py` and builds the model accordingly. At this point, you have a model with random weights that needs to be trained to output meaningful results.
```

**Source:** [models.md — Model classes](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/models.md#L50)

```text
To get a pretrained model, you need to load the weights into the model. This is done by calling [`~PreTrainedModel.from_pretrained`] which accepts weights from the Hugging Face Hub or a local directory.
```

## distractor_007

**Focus:** Disambiguate token IDs from the model-specific vocabulary they index.

**Question:** Can we freely replace a pretrained model’s tokenizer with another tokenizer just because both produce input IDs?

**Reference answer:** No. The tokenizer’s vocabulary must match the vocabulary expected by the pretrained model. A custom tokenizer with a different vocabulary cannot be assumed interchangeable merely because it produces numeric input IDs.

**Distractor:** Identical output type does not imply matching token semantics.

**Source:** [fast_tokenizers.md — Tokenizers](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/fast_tokenizers.md#L39)

```text
Whichever tokenizer you use, make sure the tokenizer vocabulary is the same as the pretrained models tokenizer vocabulary. This is especially important if you're using a custom tokenizer with a different vocabulary from the pretrained models tokenizer.
```

## distractor_008

**Focus:** Distinguish selecting one adapter from enabling adapters broadly.

**Question:** We have two attached adapters. Does enable_adapters select only the second adapter in the same way as set_adapter("lora_2")?

**Reference answer:** No. set_adapter("lora_2") selects that adapter and disables the others. enable_adapters is a broader operation that enables all adapters attached to the model.

**Distractor:** Similar method names have different selection scope.

**Source:** [peft.md — Set adapter](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/peft.md#L136)

```text
Once added, use [`~integrations.PeftAdapterMixin.set_adapter`] to force a model to use the specified adapter and disable the other adapters.
```

**Source:** [peft.md — Enable and disable adapter](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/peft.md#L144)

```text
[`~integrations.PeftAdapterMixin.enable_adapters`] is a broader function that enables *all* adapters attached to a model, and [`~integrations.PeftAdapterMixin.disable_adapters`] disables *all* attached adapters.
```

## distractor_009

**Focus:** Reject moving an inference cache optimization into training.

**Question:** Because cached attention state speeds up generation, should we also enable it during model training?

**Reference answer:** No. The documentation says caching should be used only for inference and may cause unexpected errors if enabled during training.

**Distractor:** A generation speed optimization is wrongly applied to training.

**Source:** [cache_explanation.md — Caching](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/cache_explanation.md#L29)

```text
> [!WARNING]
> Caching should only be used for **inference**. It may cause unexpected errors if it's enabled during training.
```

## distractor_010

**Focus:** Distinguish cache positions from a count of unpadded tokens.

**Question:** A reused KV cache contains ten tokens, including padding positions. Should cache_position for the next token subtract the padding count?

**Reference answer:** No. cache_position is unaffected by padding and advances once per token. For a cache containing ten tokens, the next position is torch.tensor([10]), regardless of padding tokens.

**Distractor:** Masking padding does not change the positional counter for cached tokens.

**Source:** [cache_explanation.md — Cache class](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/cache_explanation.md#L80)

```text
3. It is also important to be aware of the `cache_position`. This is important if you want to reuse a prefilled [`Cache`] with the `forward` method because you have to pass a valid `cache_position` value. This indicates the input positions in a sequence. `cache_position` is unaffected by padding, and it always adds one more position for each token. For example, if a kv cache contains 10 tokens - regardless of pad tokens - the cache position for the next token should be `torch.tensor([10])`.
```

## distractor_011

**Focus:** Distinguish dynamic growth from compile-friendly fixed capacity.

**Question:** Is the dynamically growing cache the preferred choice when we specifically need a fixed shape for compiling decoding?

**Reference answer:** No. DynamicCache grows as generation progresses. StaticCache preallocates a maximum capacity and supports compiling decoding, at the cost of unused masked positions and potentially wasted computation.

**Distractor:** Dynamic storage flexibility is confused with static-shape compilation.

**Source:** [kv_cache.md — Default cache](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L35)

```text
The [`DynamicCache`] is the default cache class for all models. It allows the cache size to grow dynamically in order to store an increasing number of keys and values as generation progresses.
```

**Source:** [kv_cache.md — Fixed-size cache](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/kv_cache.md#L72)

```text
A fixed-size cache ([`StaticCache`]) pre-allocates a specific maximum cache size for the kv pairs. You can generate up to the maximum cache size without needing to modify it. However, having a fixed (usually large) size for the key/value states means that while generating, a lot of tokens will actually be masked as they should not take part in the attention. So this trick allows to easily `compile` the decoding stage, but it incurs a waste of tokens in the attention computation. As all things, it's then a trade-off which should be very good if you generate with several sequence of more or less the same lengths, but may be sub-optimal if you have for example 1 very large sequence, and then only short sequences (as the fix cache size would be large, a lot would be wasted for the short sequences). Make sure you understand the impact if you use it!
```

## distractor_012

**Focus:** Identify exceptions to cache/graph reuse.

**Question:** If we enable a static KV cache and compile the model, can we freely change batch size or increase maximum output length without triggering recompilation?

**Reference answer:** No. Changing batch size or increasing maximum output length between calls can reinitialize and recompile the cache. generate tries to reuse the cache to avoid recompilation, so workload shape changes still matter.

**Distractor:** A static cache does not guarantee reuse across arbitrary shape changes.

**Source:** [llm_optims.md — Static kv-cache and torch.compile](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_optims.md#L71)

```text
1. If the batch size changes or the maximum output length increases between calls, the cache is reinitialized and recompiled.
2. The first several calls of the compiled function are slower because it is being compiled.
```

**Source:** [llm_optims.md — Static kv-cache and torch.compile](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/llm_optims.md#L69)

```text
Under the hood, [`~GenerationMixin.generate`] attempts to reuse the same cache object to avoid recompilation at each call, which is critical to get the most out of [torch.compile](./perf_torch_compile). Be aware of the following to avoid triggering recompilation or if generation is slower than expected.
```

## distractor_013

**Focus:** Identify a feature-triggered attention backend fallback.

**Question:** We chose SDPA but also request output_attentions=True. Should we assume the same fast attention implementation is still being used?

**Reference answer:** No. The guide lists output_attentions=True and head_mask as unsupported by SDPA and says Transformers warns and falls back to the slower eager implementation for such options.

**Distractor:** The configured backend name does not guarantee it remains active with unsupported options.

**Source:** [perf_infer_gpu_one.md — Scaled dot product attention (SDPA)](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_infer_gpu_one.md#L178)

```text
SDPA is used by default for PyTorch v2.1.1. and greater when an implementation is available. You could explicitly enable SDPA by setting `attn_implementation="sdpa"` in [`~PreTrainedModel.from_pretrained`] though. Certain attention parameters, such as `head_mask` and `output_attentions=True`, are unsupported and returns a warning that Transformers will fall back to the (slower) eager implementation.
```

## distractor_014

**Focus:** Reject an explicitly incompatible optimization combination.

**Question:** Can we combine fused AWQ modules with FlashAttention2 to stack both optimizations?

**Reference answer:** No. The AWQ documentation explicitly says fused modules cannot be combined with optimization techniques such as FlashAttention2. Choose a compatible configuration instead of enabling both together.

**Distractor:** Two individually useful optimizations are assumed compatible.

**Source:** [quantization/awq.md — Fused modules](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/awq.md#L86)

```text
> [!WARNING]
> Fused modules cannot be combined with other optimization techniques such as FlashAttention2.
```

## distractor_015

**Focus:** Separate quantization creation from quantized execution.

**Question:** Can the Marlin backend perform the initial GPTQ quantization of a full-precision model, as well as accelerate its later inference?

**Reference answer:** No. The documented Marlin backend is for quantized inference only and does not support model quantization. Initial quantization must be performed by a supported quantization workflow before Marlin is used for inference.

**Distractor:** An inference kernel is incorrectly treated as a quantizer.

**Source:** [quantization/gptq.md — Marlin](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/gptq.md#L113)

```text
[Marlin](https://github.com/IST-DASLab/marlin) is a 4-bit only CUDA GPTQ kernel, highly optimized for the NVIDIA A100 GPU (Ampere) architecture. Loading, dequantization, and execution of post-dequantized weights are highly parallelized, offering a substantial inference improvement versus the original CUDA GPTQ kernel. Marlin is only available for quantized inference and does not support model quantization.
```

## distractor_016

**Focus:** Separate quantized GPU loading from CPU offload representation.

**Question:** After enabling eight-bit CPU offloading, are the offloaded CPU weights also stored as eight-bit integers?

**Reference answer:** No. The bitsandbytes offloading section says weights dispatched to CPU are stored in float32 and are not converted to eight-bit.

**Distractor:** The model’s eight-bit label is incorrectly applied to every device’s storage.

**Source:** [quantization/bitsandbytes.md — Offloading](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L200)

```text
8-bit models can offload weights between the CPU and GPU to fit very large models into memory. The weights dispatched to the CPU are stored in **float32** and aren't converted to 8-bit. For example, enable offloading for [bigscience/bloom-1b7](https://huggingface.co/bigscience/bloom-1b7) through [`BitsAndBytesConfig`].
```

## distractor_017

**Focus:** Separate quantized weight representation from computation dtype.

**Question:** Do bnb_4bit_quant_type="nf4" and bnb_4bit_compute_dtype=torch.bfloat16 control the same thing?

**Reference answer:** No. bnb_4bit_quant_type="nf4" selects the four-bit weight quantization representation, which the guide recommends for normally distributed weights when training four-bit base models. bnb_4bit_compute_dtype selects the computation data type, such as bfloat16; it does not select the four-bit storage format.

**Distractor:** Similar quantization settings operate on different kinds of precision.

**Source:** [quantization/bitsandbytes.md — Compute data type](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L284)

```text
### Compute data type
```

**Source:** [quantization/bitsandbytes.md — Normal Float 4 (NF4)](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L297)

```text
NF4 is a 4-bit data type from the [QLoRA](https://hf.co/papers/2305.14314) paper, adapted for weights initialized from a normal distribution. You should use NF4 for training 4-bit base models.
```

**Source:** [quantization/bitsandbytes.md — Normal Float 4 (NF4)](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/bitsandbytes.md#L302)

```text
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
```

## distractor_018

**Focus:** Distinguish calibration-based methods from on-the-fly methods.

**Question:** Are GPTQ and AWQ on-the-fly methods that require no calibration dataset, just like the no-calibration approaches listed in the guide?

**Reference answer:** No. The selecting guide groups GPTQ/GPTQModel and AWQ under Calibration-based Quantization, which requires an upfront calibration step using a dataset. bitsandbytes, HQQ and torchao are listed under No Calibration Required.

**Distractor:** All reduced-precision loading workflows are assumed to have the same preparation needs.

**Source:** [quantization/selecting.md — No Calibration Required (On-the-fly Quantization)](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L35)

```text
### No Calibration Required (On-the-fly Quantization)
```

**Source:** [quantization/selecting.md — Calibration-based Quantization](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L71)

```text
These methods require an upfront calibration step using a dataset to potentially achieve higher accuracy.
```

**Source:** [quantization/selecting.md — GPTQ/GPTQModel](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L73)

```text
#### GPTQ/GPTQModel
```

**Source:** [quantization/selecting.md — AWQ (Activation-aware Weight Quantization)](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/quantization/selecting.md#L85)

```text
#### AWQ (Activation-aware Weight Quantization)
```

## distractor_019

**Focus:** Separate batch throughput opportunities from live latency constraints.

**Question:** We serve individual low-latency requests on a CPU. Should we increase Pipeline batch_size because batching is guaranteed to make inference faster?

**Reference answer:** No. Pipeline batch inference may help, especially on a GPU, but faster performance is not guaranteed. The guide advises against batching for latency-constrained live products and CPU use, and says to measure on the actual model, data and hardware.

**Distractor:** A throughput-oriented setting is treated as a universal latency improvement.

**Source:** [pipeline_tutorial.md — Batch inference](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L171)

```text
[`Pipeline`] can also process batches of inputs with the `batch_size` parameter. Batch inference may improve speed, especially on a GPU, but it isn't guaranteed. Other variables such as hardware, data, and the model itself can affect whether batch inference improves speed. For this reason, batch inference is disabled by default.
```

**Source:** [pipeline_tutorial.md — Batch inference](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/pipeline_tutorial.md#L206)

```text
1. The only way to know for sure is to measure performance on your model, data, and hardware.
2. Don't batch inference if you're constrained by latency (a live inference product for example).
3. Don't batch inference if you're using a CPU.
4. Don't batch inference if you don't know the `sequence_length` of your data. Measure performance, iteratively add to `sequence_length`, and include out-of-memory (OOM) checks to recover from failures.
5. Do batch inference if your `sequence_length` is regular, and keep pushing it until you reach an OOM error. The larger the GPU, the more helpful batch inference is.
6. Do make sure you can handle OOM errors if you decide to do batch inference.
```

## distractor_020

**Focus:** Distinguish strict single-graph compilation from permissive graph splitting.

**Question:** With fullgraph=True, will torch.compile quietly split the model into several graphs when it encounters a graph break?

**Reference answer:** No. fullgraph=True attempts to compile the entire model into one graph. torch.compile raises an error if it encounters a graph break instead of quietly accepting several graphs.

**Distractor:** The ordinary idea of graph splitting is incorrectly carried into strict fullgraph mode.

**Source:** [perf_torch_compile.md — Fullgraph](https://github.com/huggingface/transformers/blob/v4.57.1/docs/source/en/perf_torch_compile.md#L51)

```text
Fullgraph attempts to compile the entire model into a single graph to maximize performance. torch.compile raises an error if it encounters a graph break, which means it can't compile the model into a single graph.
```

