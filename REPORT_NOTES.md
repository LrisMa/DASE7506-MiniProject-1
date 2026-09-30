# MP1 Experiment and Report Notes

## Protocol and compliance

- Protocol: `7506-mp1-wt2-v2`.
- Development decisions use the validation split only. Do not use test BPB to choose a model or setting.
- All models use only the supplied training text, the supplied BPE-2048 tokenizer, and independent causal windows of 256 targets.
- The model is stateless across calls to `predict_log_probs`; causal attention prevents access to future tokens.
- Final scoring must use CPU FP32 and the same frozen checkpoint must satisfy the time, RAM, and inference-asset limits.

## Baseline

Command:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --steps 1200 --batch-size 32 --run-dir runs/baseline-7506
```

| Setting | Value |
| --- | ---: |
| Architecture | 4-layer GPT, width 128, 4 heads, GELU MLP |
| Parameters | 1,088,256 |
| Training targets | 9,830,400 |
| Seed | 17 |
| Validation BPB | 2.0710878 |
| Token perplexity | 79.5290 |
| Training time | 554.37 s |
| Validation time | 11.73 s |

## Effective mechanism: parameter-matched SwiGLU MLP

### Idea and motivation

The baseline Transformer uses a standard feed-forward sublayer:

```text
Linear(width, 4 * width) -> GELU -> Linear(4 * width, width)
```

The student model replaces only this sublayer with a gated SwiGLU form:

```text
gate, value = Linear(width, 2 * hidden)(x).chunk(2)
output = Linear(hidden, width)(SiLU(gate) * value)
```

The gate can suppress or pass candidate features depending on the current token representation. Attention, positional embeddings, causal masking, weight tying, optimizer, seed, training targets, and evaluation protocol remain unchanged. Therefore the paired comparison isolates the MLP mechanism.

### Parameter matching

For width 128, the baseline MLP has a hidden size of 512. SwiGLU uses `hidden = floor(8 * width / 3) = 341`, because it has three main matrices rather than two. The full model has 1,088,424 parameters, only 168 more than the 1,088,256-parameter baseline (about 0.015%).

### Result

Command:

```bash
python train.py --implementation student --config configs/baseline.json --device cpu --threads 4 --seed 17 --steps 1200 --batch-size 32 --run-dir runs/swiglu-7506
```

| Model | Validation BPB | Parameters | Training time | Validation time |
| --- | ---: | ---: | ---: | ---: |
| GELU baseline | 2.0710878 | 1,088,256 | 554.37 s | 11.73 s |
| Parameter-matched SwiGLU | 2.0108518 | 1,088,424 | 509.34 s | 9.95 s |

SwiGLU lowers validation BPB by 0.060236, approximately 2.9% relative to the baseline. It also has lower measured training and validation times in this run. This result is validation-only; the test split must be evaluated only after the final method is frozen.

## Ablation / negative result: weight EMA

EMA was tested with the original GELU baseline at decay 0.99. It averages training weights after step 100 and evaluates the averaged weights at the end.

| Weights evaluated | Validation BPB |
| --- | ---: |
| Raw final weights | 2.0710878 |
| EMA weights | 2.0750949 |

EMA increased BPB by 0.0040071 and increased training time from 554.37 s to 652.90 s. It is not included in the final candidate.

## Effective scaling result: medium SwiGLU model

The medium configuration in `configs/swiglu_medium.json` uses width 160, 5 heads, and depth 6. It retains the same SwiGLU mechanism and training target count as the smaller model.

| Model | Validation BPB | Parameters | Training time | Validation time |
| --- | ---: | ---: | ---: | ---: |
| Small SwiGLU (4 layers, width 128) | 2.0108518 | 1,088,424 | 509.34 s | 9.95 s |
| Medium SwiGLU (6 layers, width 160) | 1.9168978 | 2,223,992 | 994.88 s | 16.04 s |

At the same 9,830,400 training targets and seed 17, scaling lowers validation BPB by 0.093954 (about 4.7% relative to small SwiGLU). Validation time is about 1.6 times that of the small SwiGLU model, so this candidate has substantial headroom below the 5x CPU scoring limit. Final RAM and asset-size measurements remain required for the frozen checkpoint.

## Effective training-budget result: 2,400-step medium SwiGLU

The medium SwiGLU model was trained for 2,400 steps with `--eval-every 600`, processing 19,660,800 targets. Its validation curve was:

| Step | Validation BPB |
| ---: | ---: |
| 600 | 2.0820602 |
| 1,200 | 1.8638124 |
| 1,800 | 1.7782563 |
| 2,400 | 1.7475428 |

The final model used 2,223,992 parameters, trained for 1,988.37 s excluding intermediate validation, and scored validation in 16.82 s. Its BPB is 0.169355 lower than the 1,200-step medium run. Improvements diminish over time: the reductions are 0.218248, 0.085556, and 0.030713 across successive 600-step intervals.

Important comparison note: the cosine learning-rate schedule is defined over the supplied total step count. Therefore the 1,200-step point within the 2,400-step run is not a strict same-schedule comparison to the standalone 1,200-step run. Describe this result as a longer-training candidate, not as an isolated training-length ablation.

## Effective scaling result: large SwiGLU model

The large configuration in `configs/swiglu_large.json` uses width 192, 6 heads, and depth 8. It was trained for 2,400 steps with the same seed and target count as the medium 2,400-step candidate.

| Model | Validation BPB | Parameters | Training time | Validation time |
| --- | ---: | ---: | ---: | ---: |
| Medium SwiGLU, 2,400 steps | 1.7475428 | 2,223,992 | 1,988.37 s | 16.82 s |
| Large SwiGLU, 2,400 steps | 1.7131800 | 4,003,712 | 3,383.04 s | 33.47 s |

The larger model improves validation BPB by 0.034363 (about 2.0% relative), but costs about 1.7 times as much training time and twice the validation time. Its validation scoring time is about 2.85 times the baseline scoring time and remains below the 5x limit on this machine. This indicates diminishing returns from scaling alone; further work should test a complementary mechanism rather than only adding capacity.

## Effective complementary mechanism: train-only trigram interpolation

`student.py` builds a compact asset from the training split only. For each observed two-token BPE context, it stores the four most frequent next tokens and their relative frequencies. During evaluation, the asset is queried only with the current and preceding input token, then mixed with the causal neural distribution:

$$p_{final} = (1 - alpha)p_{neural} + alpha p_{trigram}.$$

The neural checkpoint is unchanged; only its prediction probabilities are mixed. The asset has 298,024 observed contexts and occupies 20.5 MiB. Together with the approximately 16 MiB large-model weights, the uncompressed inference bundle is approximately 36.5 MiB, within the 64 MiB asset limit.

| Trigram alpha | Validation BPB | Validation time |
| ---: | ---: | ---: |
| 0.00 (large neural model) | 1.7131800 | 33.47 s |
| 0.01 | 1.7000712 | 23.37 s |
| 0.02 | 1.6946916 | 22.75 s |
| 0.05 | 1.6863674 | 23.62 s |

At alpha 0.05, the mixture lowers BPB by 0.026813 relative to the frozen large neural model. The alpha is selected using validation only. Before final test scoring, retain the exact `assets/trigram_top4.pt` file alongside the final code and mixture checkpoint, and report the asset size and CPU FP32 score. Do not rebuild the asset using validation or test text.

## Negative result: frequency-adaptive trigram weight

The trigram mixer was changed to reduce its mixing weight for low-count training contexts using `alpha(c) = alpha_max * count(c) / (count(c) + tau)`. With `alpha_max = 0.10` and `tau = 10`, validation BPB was 1.688534 at 24.12 s. This improves on the pure large neural model but is 0.002167 BPB worse than fixed alpha 0.05. Therefore context frequency alone is not a useful confidence signal in this experiment; retain fixed-alpha mixing as the current best candidate.

## Effective refinement: coverage-weighted trigram mixing

The top-4 trigram entries are stored with their original training-set probability mass. Rather than treating them as a complete distribution in every context, the mixer uses `alpha_effective = alpha_max * top4_coverage`, where `top4_coverage` is the probability mass retained by the four stored continuations. This reduces n-gram influence for diffuse contexts while retaining it for concentrated local patterns.

| Alpha maximum | Validation BPB | Validation time |
| ---: | ---: | ---: |
| 0.10 | 1.6806173 | 28.03 s |
| 0.15 | 1.6769660 | 25.86 s |
| 0.20 | 1.6754425 | 26.09 s |
| 0.25 | 1.6754231 | 23.33 s |

The 0.25 candidate is selected because it has the lowest validation BPB, although its 0.000019 advantage over 0.20 is practically negligible. Coverage weighting lowers BPB by 0.010944 relative to fixed-alpha trigram mixing. No additional alpha search is justified before testing a different mechanism.

## Effective complementary mechanism: self-trained neural ensemble

The frozen 2,400-step large and medium SwiGLU models are combined by averaging their normalized neural probabilities, then applying the selected coverage-weighted trigram mixture. The first experiment gives each model weight 0.5.

| Candidate | Validation BPB | Validation time |
| --- | ---: | ---: |
| Large SwiGLU plus coverage-weighted trigram | 1.6754231 | 23.33 s |
| Large/medium 50/50 ensemble plus coverage-weighted trigram | 1.6542343 | 38.82 s |

The ensemble lowers BPB by 0.021189. Its validation time is about 3.31 times the baseline time, remaining below the 5x limit on this machine. The final bundle must include the combined ensemble checkpoint and the exact train-only trigram asset; the two neural models are packed into the ensemble checkpoint, so a separate peer checkpoint is not required for reconstruction.

The ensemble peer-weight ablation assigned the medium model weight 0.30 and the large model weight 0.70. It achieved 1.6536679 BPB at 41.28 s, only 0.000566 below the 50/50 mixture. Treat the two weights as practically tied; stop tuning this coefficient and use 0.30 only as the lowest observed validation setting.

## Effective asset-capacity refinement: top-8 trigram table

The train-only trigram asset was expanded from four to eight stored continuations per two-token context, while retaining coverage weighting and the 70/30 large/medium ensemble.

| Trigram capacity | Validation BPB | Validation time |
| --- | ---: | ---: |
| Top-4 | 1.6536679 | 41.28 s |
| Top-8 | 1.6484857 | 36.42 s |

Top-8 lowers BPB by 0.005182 and is the current best validation candidate. The measured time variation across repeated CPU evaluations should not be interpreted as a reliable speed gain; the relevant result is that both candidates remain below the 5x time limit.

## Effective within-window memory mechanism: neural cache

The cache computes similarity between the current hidden state and strictly earlier hidden states in the same evaluation window. Each earlier state votes only for the token observed immediately after it, so a prediction at position `t` uses only tokens through `t`. Cache state is rebuilt for every `predict_log_probs` call and never crosses examples or windows.

| Candidate | Validation BPB | Validation time |
| --- | ---: | ---: |
| Top-8 ensemble plus coverage-weighted trigram | 1.6484857 | 36.42 s |
| Top-8 ensemble, trigram, and cache alpha 0.05 / temperature 0.10 | 1.6100373 | 41.13 s |

The cache lowers BPB by 0.038448, the largest improvement after introducing SwiGLU. It is a strictly causal, within-window mechanism and remains below the measured 5x CPU scoring limit. The next validation-only search should vary cache weight while keeping the temperature and all other components fixed.

| Cache alpha at temperature 0.10 | Validation BPB | Validation time |
| ---: | ---: | ---: |
| 0.05 | 1.6100373 | 41.13 s |
| 0.10 | 1.6069751 | 45.47 s |
| 0.15 | 1.6103104 | 40.83 s |

Cache alpha 0.10 is selected. The 0.15 result shows that too much probability mass on the cache harms prediction, so subsequent cache experiments keep alpha fixed and tune only the cache temperature.

## Cache-temperature ablation

With the selected cache alpha of 0.10, reducing cache temperature from 0.10 to 0.05 produced validation BPB 1.6086731 at 43.00 s. Increasing it to 0.20 produced validation BPB 1.6261847 at 42.37 s. The midpoint temperature 0.075 improved validation BPB to 1.6061392 at 41.33 s, so it is the selected cache temperature. Further scalar temperature sweeps are not justified because the improvement over temperature 0.10 is only 0.0008359 BPB.

| Cache temperature at alpha 0.10 | Validation BPB | Validation time |
| ---: | ---: | ---: |
| 0.05 | 1.6086731 | 43.00 s |
| 0.10 | 1.6069751 | 45.47 s |
| 0.075 | 1.6061392 | 41.33 s |
| 0.20 | 1.6261847 | 42.37 s |

## Negative result: medium-model neural cache

The medium ensemble peer was given an independent within-window cache with alpha 0.05 and temperature 0.075, while retaining the selected large-model cache. This produced validation BPB 1.6064858, which is 0.0003466 worse than the single-cache candidate. The measured validation time was 96.36 s, above the approximately 58.64 s 5x CPU budget on this machine. This branch is excluded from the final candidate and should not receive further tuning.

## Required final-report items

- State the baseline and the parameter-matched SwiGLU comparison using the same 9,830,400 training targets.
- Include the EMA negative result as a mechanism ablation or exploratory result.
- Report seed, commands, parameter counts, training/search costs, and CPU FP32 evaluation cost.
- Disclose substantive AI assistance: the SwiGLU experiment design and implementation were developed with GitHub Copilot assistance and reviewed by the student.
- After the final method is fixed, run and report a complete test evaluation from the matching checkpoint. Do not use its score to select further changes.

## Early full-test registration result

The validation-selected pre-3200-step candidate `runs/ensemble-top8-cache-a010-t0075/checkpoint.pt` scored 1.6213424 BPB on the full test split. Its validation BPB was 1.6061392, a test-minus-validation difference of 0.0152032. The test ran concurrently with a 3200-step CPU training job and measured 116.90 s, so that timing is invalid for resource reporting. This test result is retained as an early registration and comparison point; it is not used to select subsequent methods.

## Longer-training candidate: 3,200-step large SwiGLU

The large SwiGLU architecture was retrained from random initialization for 3,200 steps, processing 26,214,400 targets. Its validation BPB was 1.6916965 at step 2,800 and 1.6832162 at step 3,200. The final step-3,200 checkpoint is selected because it has the lower validation BPB. A lower single minibatch training loss at step 3,000 is not a checkpoint-selection signal: it is noisy and no step-3,000 snapshot was saved. The selected checkpoint must next be evaluated inside the existing top-8 trigram, medium-peer ensemble, and single-cache mixture.

Replacing only the old large checkpoint in the selected top-8 trigram, 70/30 large/medium ensemble, and single-cache mixture produced validation BPB 1.5866115 in 49.45 s. This improves the previous best validation candidate by 0.0195277 BPB and remains within the approximate 5x CPU-time budget on this machine. Because the new large model is substantially stronger than its medium peer, the next validation-only check should reduce the peer weight from 0.30 to 0.20 before changing any other component.

Reducing the peer weight to 0.20 produced validation BPB 1.5881406, 0.0015291 worse than the selected 0.30 weight. An optional confidence gate then scaled cache alpha according to how concentrated its causal similarity distribution was above a uniform history baseline. This reduced validation BPB to 1.5827955 in 49.47 s, improving the selected fixed-cache candidate by 0.0038160 without increasing measured CPU time. Confidence-gated cache is the current best candidate.

## Effective second-stage continuation: mild dropout

A 400-step continuation from the selected 3,200-step large checkpoint used dropout 0.05, learning rate 0.00005, no label smoothing, and validation/checkpoint snapshots every 100 steps. Its raw large-model validation BPB improved monotonically: 1.6775554, 1.6764352, 1.6758519, and 1.6747322 at steps 100 through 400. The final checkpoint improves raw validation BPB by 0.0084841 over the original 3,200-step checkpoint. It must next be evaluated in the selected top-8 trigram, 70/30 ensemble, and confidence-gated cache mixture.

The full mixture with this checkpoint achieved validation BPB 1.5813190 in 45.42 s. This improves the prior confidence-gated-cache candidate by 0.0014765 BPB. The gain is small because the improved large neural distribution is only one component of the final ensemble/cache/trigram mixture, but it is the current validation-best candidate.

## Effective higher-order memory: frequency-pruned fourgram

A sparse fourgram table retained the top continuation for each three-token training context observed at least three times. It contains 239,403 contexts and occupies 2.7 MiB. Mixing this table after the lower-order trigram expert with maximum alpha 0.04 reduced validation BPB to 1.5775046, an improvement of 0.0038144 over the prior best. CPU validation time increased to 52.82 s, leaving limited headroom below the approximate 58.65 s budget; do not add another large inference asset.

Increasing the fourgram maximum alpha to 0.06 improved validation BPB further to 1.5764082 in 49.33 s. This is the current best candidate. One final alpha 0.08 validation check is justified by the monotonic improvement; stop this sweep after that check.

Testing the more informative higher boundary alpha 0.10 improved validation BPB to 1.5747914 in 48.42 s, the current best candidate. Because the fourgram weight is coverage-scaled by the top continuation probability, a final boundary check at alpha 0.20 is justified without changing inference cost.

Fourgram alpha 0.20 improved validation BPB further to 1.5726441 in 51.02 s. Because all tested alphas use the same model and asset, this timing variation is not treated as a speed effect. The next bounded check is alpha 0.40; stop or refine the sweep once its validation BPB stops improving.

Fourgram alpha 0.40 produced validation BPB 1.5727224, 0.0000783 worse than alpha 0.20. The final refinement is alpha 0.30; do not test larger values.

Fourgram alpha 0.30 achieved validation BPB 1.5721209 in 48.65 s, the selected fourgram setting. Further fourgram-alpha tuning is stopped. The next higher-order-memory feasibility check is a frequency-pruned top-1 fivegram asset using only four-token training contexts observed at least three times.

The fivegram asset contains 171,044 frequent contexts and occupies 1.96 MiB. With fivegram alpha 0.10 alongside the selected fourgram, trigram, ensemble, and confidence-gated cache, validation BPB improved to 1.5706621 in 54.61 s. This is the current best candidate. No further expert or asset may be added because CPU-time headroom is limited; a fivegram-alpha-only check at 0.20 does not change inference cost.

Using fivegram alpha 0.20 and increasing the confidence-gated cache alpha from 0.10 to 0.15 produced validation BPB 1.5664485 in 44.53 s, improving the previous fivegram candidate by 0.0039472. This is the current best candidate. The next bounded no-training check is cache alpha 0.20; it does not alter assets or algorithmic inference cost.

Cache alpha 0.20 improved validation BPB to 1.5646337. Alpha 0.30 produced 1.5648082, 0.0001744 worse, so cache alpha 0.20 is selected. With the cache now frozen, the next no-cost recalibration is a lower trigram alpha because the newly added fourgram and fivegram experts may reduce the value of an aggressive lower-order n-gram mixture.

Lowering the coverage-weighted trigram alpha from 0.25 to 0.15 improved validation BPB to 1.5603870 in 55.15 s, a 0.0042467 gain and the current best candidate. The next bounded check is alpha 0.05; if it worsens, refine only once at 0.10.

## Negative result: cache recency bias

The causal cache score was augmented with a recency penalty, $-lambda \log(1 + t - j)$, for earlier positions $j < t$. With `cache_recency_bias = 0.50` and all other selected settings unchanged, validation BPB increased to 1.5636108 in 49.04 s. This is 0.0032238 worse than the selected unbiased cache, so recency bias is excluded and should not be tuned further.

## Ensemble peer-weight recalibration

With the final high-order memory and confidence-gated cache settings, reducing the medium peer weight from 0.30 to 0.25 yielded validation BPB 1.5606454 in 49.05 s. This is 0.0002584 worse than the selected 0.30 value, so lower peer weights are excluded. One upper-neighbor check at 0.35 is sufficient to determine whether the optimum moved upward under the final mixture.

## Effective ensemble mechanism: geometric fusion

The arithmetic ensemble averages model probabilities. A geometric alternative combines log-probabilities and renormalizes: $\log p = \operatorname{logsoftmax}((1-w)\log p_{large} + w\log p_{medium})$. With the same peer weight 0.30 and all other selected components fixed, geometric fusion achieved validation BPB 1.5580391 in 42.66 s, improving the arithmetic ensemble result by 0.0023479. This is the current best validation candidate and introduces no additional inference assets.

Temperature-sharpening the geometric ensemble with `ensemble_temperature = 0.90` increased validation BPB to 1.5745676 in 45.03 s, a 0.0165286 regression. The geometric ensemble is already appropriately calibrated at temperature 1.0; do not apply further temperature scaling.

## Frozen final test registration

The validation-selected geometric ensemble checkpoint `runs/final-candidate-geometric-ensemble/checkpoint.pt` was evaluated once on the test split after selection was frozen. It achieved test BPB 1.5741006, token perplexity 26.8580, and CPU FP32 evaluation time 50.05 s. The checkpoint SHA-256 was `ba7a9e73f3b2b87ab2fc8148ba6f07944ffe3da86959e803fa1a27dd115ccee6`.

The required uncompressed inference bundle consists of the checkpoint plus `assets/trigram_top8.pt`, `assets/fourgram_top1_min3.pt`, and `assets/fivegram_top1_min3.pt`. Its measured size is 57,406,488 bytes (54.75 MiB), below the 64 MiB limit.