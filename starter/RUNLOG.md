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