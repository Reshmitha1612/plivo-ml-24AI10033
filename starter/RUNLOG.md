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