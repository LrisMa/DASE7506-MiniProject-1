# DASE7506 Mini Project 1: A Compact Causal Ensemble for WikiText-2

Student: MA Yumeng(马钰萌)  ID: 3036780527
Repository: https://github.com/LrisMa/DASE7506-MiniProject-1  
Protocol: `7506-mp1-wt2-v2`

## Abstract

This project improves the supplied small GPT baseline for next-token prediction on the supplied WikiText-2 text and fixed BPE-2048 tokenizer. The final system combines a SwiGLU Transformer, a self-trained medium-model peer, a causal within-window neural cache, and compact train-only n-gram memories. All development choices were selected on validation only. The frozen checkpoint achieved **1.5741006 test BPB** under CPU FP32 evaluation in 49.25--50.05 seconds. The uncompressed inference bundle is 54.75 MiB, below the 64 MiB limit.

## 1. Task, Protocol, and Evaluation

The task is to minimize bits per UTF-8 byte (BPB) on WikiText-2 while preserving causality and the supplied evaluation protocol. The tokenizer has a fixed vocabulary of 2,048 BPE tokens and evaluation uses independent windows of 256 targets. I used only the supplied training split to train parameters and construct lookup assets. I used the validation split for model selection and evaluated the test split only after freezing the final candidate.

The final evaluator receives normalized natural log-probabilities of shape `[batch, time, 2048]`. CPU FP32 was used for the registered result. No pretrained weights, external text, test-set tuning, cross-window state, or future-token access was used.

## 2. Baseline and Controlled Architecture Comparison

The supplied baseline is a four-layer causal GPT with width 128, four heads, GELU feed-forward layers, tied token/output embeddings, and 1,088,256 parameters. It was trained for 1,200 updates of batch size 32 and context 256, i.e. 9,830,400 next-token targets, using seed 17.

My first controlled change replaces only the GELU feed-forward layer with SwiGLU. A standard baseline MLP uses

$$
\mathrm{Linear}(d,4d) \rightarrow \mathrm{GELU} \rightarrow \mathrm{Linear}(4d,d).
$$

For SwiGLU, I use `hidden = floor(8d/3)` so that the extra gate matrix is parameter matched:

$$
g,v=\mathrm{Linear}(d,2h)(x), \qquad \mathrm{MLP}(x)=\mathrm{Linear}(h,d)(\mathrm{SiLU}(g) \odot v).
$$

All other conditions, including seed, training targets, attention, optimizer, tokenizer, and evaluation protocol, were unchanged. This isolates the architectural change.

| Model | Parameters | Training targets | Validation BPB | Train time | Validation time |
|---|---:|---:|---:|---:|---:|
| Supplied GELU GPT | 1,088,256 | 9,830,400 | 2.0710878 | 554.37 s | 11.73 s |
| Parameter-matched SwiGLU GPT | 1,088,424 | 9,830,400 | 2.0108518 | 509.34 s | 9.95 s |

SwiGLU lowers validation BPB by 0.060236 (2.9% relative) with only 168 extra parameters. This paired result satisfies the equal-training-target comparison required by the assignment.

## 3. Model Scaling and Training

I scaled the SwiGLU architecture while keeping the supplied tokenizer and causal attention. The medium configuration uses width 160, five heads, and six layers. The large configuration uses width 192, six heads, and eight layers. Longer training was useful but showed diminishing returns.

| Model | Updates | Parameters | Validation BPB | Training time |
|---|---:|---:|---:|---:|
| Medium SwiGLU | 2,400 | 2,223,992 | 1.7475428 | 1,988.37 s |
| Large SwiGLU | 2,400 | 4,003,712 | 1.7131800 | 3,383.04 s |
| Large SwiGLU | 3,200 | 4,003,712 | 1.6832162 | -- |

The 3,200-step large model processes 26,214,400 targets. I then ran a 400-step mild continuation from this checkpoint using dropout 0.05 and learning rate 0.00005. Its standalone validation BPB reached 1.6747322. This continuation was selected before later mixture experiments.

An exponential moving average of the baseline weights was an unsuccessful alternative: it increased validation BPB from 2.0710878 to 2.0750949 and increased training time. I therefore did not include EMA in the final system.

## 4. Final Predictor

The final predictor starts from the large SwiGLU distribution and combines four complementary sources. The frozen large checkpoint is paired with a 2,400-step medium SwiGLU checkpoint at peer weight 0.30. Rather than arithmetic probability averaging, the final version uses geometric fusion:

$$
\log p_{ens}=\mathrm{logsoftmax}(0.70\log p_{large}+0.30\log p_{medium}).
$$

This emphasizes continuations supported by both independently trained models.

### 4.1 Causal within-window neural cache

For a position $t$, normalized hidden state $h_t$ is compared with each earlier state $h_j$, $j<t$. The cache turns each earlier state into a vote for the observed next token $x_{j+1}$. Scores are masked for $j \geq t$, so the cache cannot read the current or future target. Cache state is local to one call of `predict_log_probs`, so it cannot carry information across windows or examples.

The cache distribution is interpolated with the neural distribution using a maximum weight of 0.20 and temperature 0.075. A confidence gate reduces this weight when its similarity distribution is diffuse. This is the strongest single complementary mechanism found after the neural architecture: on the earlier top-8 ensemble, cache addition reduced validation BPB from 1.6484857 to 1.6100373.

### 4.2 Train-only n-gram memories

Three compact tables are built exclusively from the training tokens:

- A top-8 trigram table uses the two preceding BPE tokens.
- A top-1 fourgram table uses three preceding tokens and retains contexts observed at least three times.
- A top-1 fivegram table uses four preceding tokens and retains contexts observed at least three times.

Each table supplies only a probability distribution conditioned on an already observed prefix. The tables are interpolated after the neural/cache distribution. The selected weights are 0.15 for coverage-weighted trigram, 0.30 for fourgram, and 0.20 for fivegram. This preserves causality and does not use validation or test text.

## 5. Ablations and Analysis

Table 3 summarizes the selected validation path. Timings vary modestly between CPU runs; BPB is the selection measure, while timing is checked against the resource bound.

| Added or changed component | Validation BPB | Validation time |
|---|---:|---:|
| Large/medium arithmetic ensemble + top-8 trigram | 1.6484857 | 36.42 s |
| Add causal cache | 1.6100373 | 41.13 s |
| Cache temperature 0.075 | 1.6061392 | 41.33 s |
| 3,200-step large model + confidence-gated cache | 1.5827955 | 49.47 s |
| Mild-dropout large continuation | 1.5813190 | 45.42 s |
| Add selected fourgram | 1.5721209 | 48.65 s |
| Add selected fivegram and cache recalibration | 1.5646337 | -- |
| Lower trigram weight to 0.15 | 1.5603870 | 55.15 s |
| Geometric rather than arithmetic ensemble fusion | **1.5580391** | 42.66 s |

The final geometric fusion improves validation BPB by 0.0023479 relative to arithmetic fusion. Its rationale is that the large and medium models have different capacity and training trajectories; their agreement is a useful confidence signal. However, the gain is small, so it should be interpreted as a calibration refinement rather than a new model class.

I also report negative results to limit overclaiming:

| Experiment | Validation BPB | Interpretation |
|---|---:|---|
| EMA, decay 0.99 | 2.0750949 | Worse than the raw baseline. |
| Frequency-adaptive trigram | 1.6885340 | Worse than fixed trigram mixing. |
| Medium peer cache | 1.6064858 | Slightly worse and 96.36 s, above the time limit. |
| Cache recency bias 0.50 | 1.5636108 | Worse than the selected unbiased cache. |
| Geometric-fusion temperature 0.90 | 1.5745676 | Excessive sharpening severely hurts calibration. |

These failures suggest that useful retrieval-like information must be locally relevant and carefully calibrated. In particular, naive sharpening and extra cache complexity can over-concentrate the predictive distribution.

## 6. Final Test Result and Resource Compliance

After freezing validation choices, I evaluated `runs/final-candidate-geometric-ensemble/checkpoint.pt` once on the complete test split. A separate fresh GitHub clone reproduced the same result and passed all five contract tests.

| Metric | Final value |
|---|---:|
| Test BPB | **1.5741006374** |
| Test token perplexity | 26.8580004 |
| CPU FP32 test time | 49.25 s (initial run: 50.05 s) |
| Baseline CPU validation time | 11.73 s |
| Approximate 5x time budget | 58.64 s |
| Peak allocated/reserved accelerator memory | 0.0 GiB on CPU |
| Uncompressed inference bundle | 57,406,488 bytes (54.75 MiB) |

The bundle contains the final checkpoint plus `trigram_top8.pt`, `fourgram_top1_min3.pt`, and `fivegram_top1_min3.pt`. Thus it is below the 64 MiB asset limit. The checkpoint SHA-256 is `ba7a9e73f3b2b87ab2fc8148ba6f07944ffe3da86959e803fa1a27dd115ccee6`.

The main limitation is evaluation cost: the final model is close to the 5x CPU limit, leaving little headroom for additional experts. Moreover, the validation-to-test gap is expected on a finite development set. The frozen test score was not used to select any further setting.

## 7. Reproducibility and AI Disclosure

The repository README contains installation, asset, checkpoint, and CPU FP32 evaluation commands. The final checkpoint and its three required assets are supplied separately as a matching bundle. Running

```bash
python evaluate.py \
  --checkpoint runs/final-candidate-geometric-ensemble/checkpoint.pt \
  --device cpu --precision fp32 --split test
```

reproduces the registered score when run from `code/` with the bundle extracted there.

GitHub Copilot provided substantive assistance with experiment design, implementation, debugging, and report-note drafting. I reviewed the implementation, chose experiments based on validation results, executed the reported commands, and take responsibility for the submitted work.

## References

1. S. Merity, C. Xiong, J. Bradbury, and R. Socher. *Pointer Sentinel Mixture Models*. ICLR, 2017. WikiText-2 source dataset.
2. N. Shazeer. *GLU Variants Improve Transformer*. arXiv:2002.05202, 2020.
3. Assignment materials: DASE7506 MP1 guide and supplied starter code, 2026.