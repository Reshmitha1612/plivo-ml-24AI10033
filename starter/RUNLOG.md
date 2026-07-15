# Run 0 (Baseline)

Hypothesis:
Establishing a baseline using the provided implementation.

Changes:
None.

Training:
2000 optimizer steps.

Parameters:
1,339,840

Final training loss:
1.7315

Dev BPB:
2.3718

Conclusion:
Baseline established.We will compare future runs against this score.


# Run 1 (Weight Tying)

Hypothesis:
Sharing the input embedding and output projection weights may improve generalization while reducing the number of trainable parameters, as this technique is commonly used in language models.

Changes:
Enabled weight tying by setting `tie_weights = True` in `model.py`.

Training:
2000 optimizer steps.

Parameters:
1,298,880

Final training loss:
1.7651

Dev BPB:
2.4122

Conclusion:
Weight tying reduced the parameter count by around 41K, but the development BPB increased from 2.3718 to 2.4122, indicating worse performance. Under the 2000-step training budget, the model likely benefited from having separate input and output embeddings. This change was reverted for subsequent experiments.


# Run 2 (AdamW Optimizer)

Hypothesis:
AdamW may improve generalization by decoupling weight decay from the adaptive parameter updates compared to the baseline Adam optimizer.

Changes:
Replaced the Adam optimizer with AdamW using a weight decay of 0.01. All other settings remained unchanged.

Training:
2000 optimizer steps.

Parameters:
1,339,840

Final training loss:
1.7322

Dev BPB:
2.3723

Conclusion:
The optimizer change produced almost identical performance to the baseline, with a slightly higher BPB (2.3723 vs. 2.3718). Under the 2000-step training budget, AdamW alone did not provide a measurable benefit, so the baseline Adam optimizer was retained for subsequent experiments.


# Run 3 (Gradient Clipping)

Hypothesis:
The baseline trainer does not use gradient clipping, which may allow occasional large gradient updates during optimization. Clipping the gradient norm could stabilize training and improve generalization within the limited 2000-step budget.

Changes:
Added gradient clipping using:

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

Training:
2000 optimizer steps.

Parameters:
1,339,840

Final training loss:
1.7133

Dev BPB:
2.3526

Conclusion:
Gradient clipping consistently improved both the final training loss and the development BPB compared to the baseline. This suggests that limiting large gradient updates leads to more stable optimization under the fixed training budget. This change will be retained for future experiments.

# Run 4 (Learning Rate Schedule)

Hypothesis:
A linear warmup followed by cosine learning-rate decay would improve optimization under the fixed 2000-step training budget compared to a constant learning rate.

Changes:
Retained the Adam optimizer and gradient clipping from the previous best configuration. Added a learning-rate schedule consisting of a 100-step linear warmup followed by cosine decay.

Training:
2000 optimizer steps.

Parameters:
1,339,840

Final training loss:
1.9020

Dev BPB:
2.6066

Conclusion:
The learning-rate schedule substantially degraded performance compared to the previous best run (2.6066 vs. 2.3526 BPB). With only 2000 optimizer steps, the warmup phase likely reduced the effective optimization time too much, preventing the model from reaching the same quality as the constant learning-rate baseline. This change was reverted.

# Run 5 (Model Capacity Scaling)

Hypothesis:
The previous best configuration (Run 3) only utilized ~1.34M of the allowed 2.0M parameter budget. By increasing the hidden dimension, we can increase the model's representational capacity and better utilize the parameter cap to improve generalization.

Changes:
In model.py, increased `n_embd` from 160 to 192, and adjusted `n_head` from 4 to 6 to maintain clean division. All other settings (baseline Adam + gradient clipping) remained unchanged.

Training:
2000 optimizer steps.

Parameters:
1,902,720

Final training loss:
1.6841

Dev BPB:
2.3408

Conclusion:
Scaling the model utilized the remaining parameter budget effectively, leading to a new best BPB (down from 2.3526). The wider network absorbed more features within the 2000 steps without overfitting. This configuration will be kept as the new baseline.

# Run 6 (Weight Tying to Create Parameter Headroom)

Hypothesis:
We need to expand the tokenizer vocabulary to handle the Devanagari (Hindi) text efficiently, but the previous configuration is at 1.90M parameters. Expanding the vocabulary to ~1,000 tokens without tied weights would cost roughly 384K parameters and push us well over the 2.0M cap. By turning on weight tying, we can deliberately free up parameters to create the headroom needed for this upcoming tokenizer upgrade.

Changes:
In model.py, changed `tie_weights` from False to True. All other settings remained unchanged.

Training:
2000 optimizer steps.

Parameters:
1,853,568

Final training loss:
1.7168

Dev BPB:
2.3722

Conclusion:
As expected, performance degraded slightly (BPB increased from 2.3408 to 2.3722) because forcing the input and output embeddings to share weights reduces the model's degrees of freedom. However, this was a necessary strategic trade-off: it freed up ~49K parameters, dropping our total to 1.85M. We will KEEP this configuration, as we now have the exact budget required to expand our tokenizer vocabulary to 1,000 tokens in the next run without violating the 2M hard cap.


# Run 7 (BPE Tokenizer Implementation)

Hypothesis:
The baseline byte-level tokenizer forces Devanagari (Hindi) characters to consume 3 tokens each, artificially inflating sequence lengths and crippling the model's effective context window. Training a custom Byte-Pair Encoding (BPE) tokenizer to expand the vocabulary should compress the text sequence, allowing the model to process significantly more actual content within its 128-token context window per step.

Changes:
Replaced the byte-level tokenizer with a custom BPE tokenizer trained directly on `train_corpus.txt`. Expanded `vocab_size` to 1000. (The weight tying enabled in Run 6 kept the embedding parameter growth within the 2.0M cap).

Training:
2000 optimizer steps.

Parameters:
1,996,416

Final training loss:
4.1939

Dev BPB:
2.1751

Conclusion:
A good result. The dev BPB dropped drastically from 2.3722 to 2.1751. The tokenizer compressed the 7.3MB corpus from ~7.3M tokens down to 2.6M tokens. This allowed the model to effectively see ~2.8x more context per optimization step. The parameter count is perfectly maximized at 99.8% of the budget. We will keep this configuration as the new baseline.