# DeepCAD Latent Rescue Snapshot (2026-04-16)

## Purpose

This file fixes the important experimental facts for the current rescue route so later iterations
(especially text-modality work) do not overwrite the current conclusions.

## Datasets

### Baseline training set

- ids: `datasets/rescue_deepcad_latent/overlap_deep_first_len60_trainplusval/train_ids.txt`
- size: `117,497`
- test ids: `datasets/rescue_deepcad_latent/overlap_deep_first_len60_trainplusval/test_ids.txt`
- test size: `5,592`
- train latent root: `datasets/rescue_deepcad_latent/latents/overlap_deep_first_len60_trainplusval_fp16/train`
- test latent root: `datasets/rescue_deepcad_latent/latents/overlap_deep_first_len60_trainplusval_fp16/test`

### Expanded full-v0 training set

- ids: `datasets/rescue_deepcad_latent/full_v0_len60_excluding_rescue_test/train_ids.txt`
- size: `318,886`
- test ids: `datasets/rescue_deepcad_latent/full_v0_len60_excluding_rescue_test/test_ids.txt`
- test size: `5,592`
- train latent root: `datasets/rescue_deepcad_latent/latents/full_v0_len60_excluding_rescue_test_fp16/train`
- test latent root: `datasets/rescue_deepcad_latent/latents/overlap_deep_first_len60_trainplusval_fp16/test`

Note:
- The held-out test is intentionally the same rescue overlap test as before.
- `full_v0_len60_excluding_rescue_test` removes any sample whose base id overlaps that held-out test.

## Training Runs

### Run A: overlap-only baseline

- run dir: `runs/deepcad_latent/resnet18_gru_v1_ddp`
- checkpoint: `best.pt`
- best epoch: `6`
- backbone: `resnet18`
- train batch size per rank: `64`
- num workers per rank: `4`
- epochs requested: `30`
- lr: `1e-4`
- weight decay: `1e-4`

### Run B: expanded full-v0 train set

- run dir: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2`
- checkpoint: `best.pt`
- best epoch: `7`
- backbone: `resnet18`
- train batch size per rank: `64`
- num workers per rank: `8`
- epochs requested: `12`
- lr: `1e-4`
- weight decay: `1e-4`

### Run C: expanded full-v0 image+text train set

- run dir: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_v1_ddp2`
- checkpoint: `best.pt`
- best epoch: `2`
- backbone: `resnet18`
- text encoder: `bert-base-uncased` frozen, precomputed embeddings
- train batch size per rank: `64`
- num workers per rank: `8`
- epochs requested: `10`
- freeze image epochs: `2`
- text dropout: `0.3`
- lr: `5e-5`
- weight decay: `1e-4`
- init image checkpoint: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/best.pt`
- train text root: `datasets/rescue_deepcad_latent/text_emb/full_v0_len60_excluding_rescue_test_bert_base_uncased/train`
- test text root: `datasets/rescue_deepcad_latent/text_emb/full_v0_len60_excluding_rescue_test_bert_base_uncased/test`

### Run D: expanded full-v0 image+text train set, frozen image branch

- run dir: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2`
- checkpoint: `best.pt`
- best epoch: `10`
- backbone: `resnet18`
- text encoder: `bert-base-uncased` frozen, precomputed embeddings
- train batch size per rank: `64`
- num workers per rank: `8`
- epochs requested: `10`
- freeze image epochs: `10`
- text dropout: `0.3`
- lr: `5e-5`
- weight decay: `1e-4`
- init image checkpoint: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/best.pt`
- train text root: `datasets/rescue_deepcad_latent/text_emb/full_v0_len60_excluding_rescue_test_bert_base_uncased/train`
- test text root: `datasets/rescue_deepcad_latent/text_emb/full_v0_len60_excluding_rescue_test_bert_base_uncased/test`

### Run E: expanded full-v0 image-only transformer train set

- run dir: `runs/deepcad_latent/resnet18_transformer_v1_ddp2`
- checkpoint: `best.pt`
- best epoch: `11`
- backbone: `resnet18`
- view fusion: `transformer`
- transformer layers: `2`
- transformer heads: `8`
- transformer dropout: `0.1`
- train batch size per rank: `64`
- num workers per rank: `8`
- epochs requested: `12`
- lr: `1e-4`
- weight decay: `1e-4`

### Run F: expanded full-v0 image+text transformer train set, frozen image branch

- run dir: `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2`
- checkpoint: `best.pt`
- best epoch: `10`
- backbone: `resnet18`
- view fusion: `transformer`
- transformer layers: `2`
- transformer heads: `8`
- transformer dropout: `0.1`
- text encoder: `bert-base-uncased` frozen, precomputed embeddings
- train batch size per rank: `64`
- num workers per rank: `8`
- epochs requested: `10`
- freeze image epochs: `10`
- text dropout: `0.3`
- lr: `5e-5`
- weight decay: `1e-4`
- init image checkpoint: `runs/deepcad_latent/resnet18_transformer_v1_ddp2/best.pt`
- train text root: `datasets/rescue_deepcad_latent/text_emb/full_v0_len60_excluding_rescue_test_bert_base_uncased/train`
- test text root: `datasets/rescue_deepcad_latent/text_emb/full_v0_len60_excluding_rescue_test_bert_base_uncased/test`

## Evaluation Files

### Run A

- direct: `runs/deepcad_latent/resnet18_gru_v1_ddp/eval_test_best.json`
- direct + solid: `runs/deepcad_latent/resnet18_gru_v1_ddp/eval_test_best_solid.json`
- nearest: `runs/deepcad_latent/resnet18_gru_v1_ddp/eval_test_best_nearest.json`
- blend a=0.5: `runs/deepcad_latent/resnet18_gru_v1_ddp/eval_test_best_blend_a05.json`
- blend a=0.7: `runs/deepcad_latent/resnet18_gru_v1_ddp/eval_test_best_blend_a07.json`

### Run B

- direct: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_test_best.json`
- direct + solid: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_test_best_solid.json`
- nearest: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_test_best_nearest.json`
- blend a=0.5: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_test_best_blend_a05.json`
- blend a=0.7: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_test_best_blend_a07.json`
- repeated old-overlap direct check: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_oldoverlap_test_best.json`
- repeated old-overlap nearest check: `runs/deepcad_latent/resnet18_gru_fullv0len60_v1_ddp2/eval_oldoverlap_test_best_nearest.json`

### Run C

- direct: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_v1_ddp2/eval_test_best.json`
- direct + solid: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_v1_ddp2/eval_test_best_solid.json`
- blend a=0.5: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_v1_ddp2/eval_test_best_blend_a05.json`

### Run D

- direct: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best.json`
- direct + solid: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_solid.json`
- direct + paper metrics: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_paper_metrics.json`
- direct + paper metrics + CD: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_paper_metrics_cd.json`
- blend a=0.5 + solid: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_blend_a05_solid.json`
- blend a=0.5 + paper metrics + CD: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_test_best_blend_a05_paper_metrics_cd.json`
- ARE-reference direct: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_are_reference_direct.json`
- ARE-reference blend a=0.5: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_are_reference_blend_a05.json`
- ARE-reference direct vs decoded latent + AE ceiling: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_are_reference_direct_vs_decoded_latent.json`
- ARE-reference blend a=0.5 vs decoded latent + AE ceiling: `runs/deepcad_latent/resnet18_gru_text_fullv0len60_frozenimg_v1_ddp2/eval_are_reference_blend_a05_vs_decoded_latent.json`

### Run E

- direct + paper metrics + CD: `runs/deepcad_latent/resnet18_transformer_v1_ddp2/eval_test_best_paper_metrics_cd.json`
- blend a=0.5 + paper metrics + CD: `runs/deepcad_latent/resnet18_transformer_v1_ddp2/eval_test_best_blend_a05_paper_metrics_cd.json`

### Run F

- direct + paper metrics + CD: `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2/eval_test_best_paper_metrics_cd.json`
- blend a=0.5 + paper metrics + CD: `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2/eval_test_best_blend_a05_paper_metrics_cd.json`
- ARE-reference direct + paper metrics + CD: `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2/eval_are_reference_direct_paper_metrics_cd.json`
- ARE-reference blend a=0.5 + paper metrics + CD: `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2/eval_are_reference_blend_a05_paper_metrics_cd.json`

## Key Results

### Run A direct

- `cmd_token_acc = 0.8212`
- `token_exact_acc = 0.5094`
- `sequence_cmd_exact_rate = 0.6327`
- `sequence_exact_rate = 0.1232`
- `pred_solid_valid_rate = 0.7632`

### Run A nearest

- `cmd_token_acc = 0.8194`
- `token_exact_acc = 0.5400`
- `sequence_cmd_exact_rate = 0.6794`
- `sequence_exact_rate = 0.1677`

### Run A blend alpha=0.5

- `cmd_token_acc = 0.8202`
- `token_exact_acc = 0.5389`
- `sequence_cmd_exact_rate = 0.6767`
- `sequence_exact_rate = 0.1615`

### Run B direct

- `cmd_token_acc = 0.8081`
- `token_exact_acc = 0.5161`
- `sequence_cmd_exact_rate = 0.6193`
- `sequence_exact_rate = 0.1305`
- `pred_solid_valid_rate = 0.7913`

### Run B nearest

- `cmd_token_acc = 0.8247`
- `token_exact_acc = 0.5506`
- `sequence_cmd_exact_rate = 0.6892`
- `sequence_exact_rate = 0.1735`

### Run B blend alpha=0.5

- `cmd_token_acc = 0.8258`
- `token_exact_acc = 0.5495`
- `sequence_cmd_exact_rate = 0.6854`
- `sequence_exact_rate = 0.1683`

### Run B blend alpha=0.7

- `cmd_token_acc = 0.8233`
- `token_exact_acc = 0.5432`
- `sequence_cmd_exact_rate = 0.6649`
- `sequence_exact_rate = 0.1590`

### Run C direct

- `cmd_token_acc = 0.8319`
- `token_exact_acc = 0.5327`
- `sequence_cmd_exact_rate = 0.6615`
- `sequence_exact_rate = 0.1411`
- `mean_len_abs_error = 1.7845`

### Run C blend alpha=0.5

- `cmd_token_acc = 0.8349`
- `token_exact_acc = 0.5570`
- `sequence_cmd_exact_rate = 0.6987`
- `sequence_exact_rate = 0.1760`
- `mean_len_abs_error = 1.8822`

### Run D direct

- `cmd_token_acc = 0.8365`
- `token_exact_acc = 0.5369`
- `sequence_cmd_exact_rate = 0.6685`
- `sequence_exact_rate = 0.1457`
- `pred_solid_valid_rate = 0.8308`
- `acc_cmd = 0.8653`
- `acc_param = 0.7368`
- `invalidity_ratio = 0.1692`
- `cd_mean = 0.1455`
- `cd_median = 0.04915`

### Run D blend alpha=0.5

- `cmd_token_acc = 0.8398`
- `token_exact_acc = 0.5640`
- `sequence_cmd_exact_rate = 0.7044`
- `sequence_exact_rate = 0.1786`
- `pred_solid_valid_rate = 0.9456`
- `acc_cmd = 0.8528`
- `acc_param = 0.7635`
- `invalidity_ratio = 0.0544`
- `cd_mean = 0.1338`
- `cd_median = 0.04522`

### Run E direct

- `cmd_token_acc = 0.8391`
- `token_exact_acc = 0.5305`
- `sequence_cmd_exact_rate = 0.6751`
- `sequence_exact_rate = 0.1313`
- `pred_solid_valid_rate = 0.8135`
- `acc_cmd = 0.8634`
- `acc_param = 0.7415`
- `invalidity_ratio = 0.1865`
- `cd_mean = 0.1298`
- `cd_median = 0.03433`

### Run E blend alpha=0.5

- `cmd_token_acc = 0.8406`
- `token_exact_acc = 0.5488`
- `sequence_cmd_exact_rate = 0.7030`
- `sequence_exact_rate = 0.1579`
- `pred_solid_valid_rate = 0.9442`
- `acc_cmd = 0.8528`
- `acc_param = 0.7587`
- `invalidity_ratio = 0.0558`
- `cd_mean = 0.1257`
- `cd_median = 0.04295`

### Run F direct

- `cmd_token_acc = 0.8468`
- `token_exact_acc = 0.5478`
- `sequence_cmd_exact_rate = 0.6879`
- `sequence_exact_rate = 0.1423`
- `pred_solid_valid_rate = 0.8450`
- `acc_cmd = 0.8721`
- `acc_param = 0.7528`
- `invalidity_ratio = 0.1550`
- `cd_mean = 0.1294`
- `cd_median = 0.02992`

### Run F blend alpha=0.5

- `cmd_token_acc = 0.8498`
- `token_exact_acc = 0.5690`
- `sequence_cmd_exact_rate = 0.7216`
- `sequence_exact_rate = 0.1742`
- `pred_solid_valid_rate = 0.9464`
- `acc_cmd = 0.8620`
- `acc_param = 0.7720`
- `invalidity_ratio = 0.0536`
- `cd_mean = 0.1251`
- `cd_median = 0.03365`

### ARE-reference subset (DeepCAD official test overlap, original-version only)

- ids: `datasets/rescue_deepcad_latent/are_reference_test/test_ids.txt`
- size: `5,231`
- construction: filter `overlap_deep_first_len60_trainplusval/test_ids.txt` to samples ending in `_00001`

#### Run D direct on ARE-reference subset

- `acc_cmd = 0.8701`
- `acc_param = 0.7413`
- `invalidity_ratio = 0.1665`
- `cd_mean = 0.1426`
- `cd_median = 0.04687`

#### Run D blend alpha=0.5 on ARE-reference subset

- `acc_cmd = 0.8590`
- `acc_param = 0.7674`
- `invalidity_ratio = 0.0535`
- `cd_mean = 0.1313`
- `cd_median = 0.04299`

#### AE ceiling on ARE-reference subset (`CAD_z_true` vs `CAD_true`)

- `acc_cmd = 0.9967`
- `acc_param = 0.9649`
- `invalidity_ratio = 0.0656`
- `cd_mean = 0.00448`
- `cd_median = 0.000626`
- `sequence_exact_rate = 0.8583`

#### Run F direct on ARE-reference subset

- `acc_cmd = 0.8775`
- `acc_param = 0.7577`
- `invalidity_ratio = 0.1520`
- `cd_mean = 0.1265`
- `cd_median = 0.02712`

#### Run F blend alpha=0.5 on ARE-reference subset

- `acc_cmd = 0.8680`
- `acc_param = 0.7766`
- `invalidity_ratio = 0.0522`
- `cd_mean = 0.1225`
- `cd_median = 0.03090`

## Latent Structure Analysis

### Global family statistics

- analysis file: `runs/deepcad_latent/latent_structure_analysis_v1.json`
- focused family file: `runs/deepcad_latent/latent_family_focus_v1.json`
- scanned ids: `318,886` (`full_v0_len60_excluding_rescue_test/train_ids.txt`)
- non-empty topology families discovered: `48,427`

Most common families:

- `Line -> Line -> Line -> Line -> Ext`: `58,703`
- `Circle -> Ext`: `23,552`
- `Circle -> Circle -> Ext`: `14,145`
- `Line -> Line -> Line -> Line -> Ext -> Line -> Line -> Line -> Line -> Ext`: `10,431`
- `Line -> Line -> Line -> Line -> Line -> Line -> Line -> Line -> Ext`: `8,996`

Interpretation:

- The CAD sequence distribution is highly long-tailed.
- A relatively small number of topology families dominates the dataset.
- This supports the empirical observation that simple/common CAD structures are much easier for the image/text-to-latent model than long, rare, multi-extrusion structures.

### Family-level latent geometry

Within a fixed topology family, latent distances correlate moderately with parameter distances:

- `Line -> Line -> Line -> Line -> Ext`: `corr = 0.367`
- `Circle -> Ext`: `corr = 0.476`
- `Circle -> Circle -> Ext`: `corr = 0.383`
- `Line x8 -> Ext`: `corr = 0.439`
- `Circle -> Ext -> Circle -> Ext`: `corr = 0.425`

Linear recoverability of parameters from latent vectors (`z -> params`, family-wise linear regression):

- `Line -> Line -> Line -> Line -> Ext`: `R² = 0.603`
- `Circle -> Ext`: `R² = 0.505`
- `Circle -> Circle -> Ext`: `R² = 0.312`
- `Line -> Line -> Line -> Line -> Ext -> Line -> Line -> Line -> Line -> Ext`: `R² = 0.318`
- `Circle -> Ext -> Circle -> Ext`: `R² = 0.270`

Interpretation:

- The pretrained DeepCAD latent space is not arbitrary.
- Within simple topology families, continuous geometry is partially preserved in a locally structured way.
- This structure becomes substantially less linear / less recoverable for more complex multi-extrusion families.

### Focused family findings

#### `Circle -> Ext`

Key parameter recoverability:

- `circle1_radius`: `R² = 0.986`
- `ext1_plane_gamma`: `R² = 0.978`
- `ext1_plane_theta`: `R² = 0.978`
- `ext1_plane_phi`: `R² = 0.970`
- `ext1_sketch_size`: `R² = 0.868`
- `ext1_sketch_pos_x`: `R² = 0.795`

Dominant latent directions:

- `PC1` strongly correlates with `ext1_sketch_size` (`0.754`) and `ext1_sketch_pos_x` (`-0.719`)
- `PC1` also correlates with plane orientation (`plane_phi ≈ 0.509`, `plane_gamma/theta ≈ -0.508`)

Interpretation:

- In this simplest cylindrical family, latent vectors clearly encode radius, sketch scale and plane orientation.
- This family is a strong candidate for visualization in the thesis because the geometry-to-latent relationship is very clean.

#### `Circle -> Circle -> Ext`

Key parameter recoverability:

- `circle1_radius`: `R² = 0.978`
- `ext1_plane_gamma/theta`: `R² = 0.967`
- `ext1_plane_phi`: `R² = 0.955`
- `ext1_sketch_size`: `R² = 0.896`
- `circle2_radius`: `R² = 0.888`
- `ext1_sketch_pos_x`: `R² = 0.850`

Interpretation:

- Even with a second circular primitive, the latent still preserves major continuous geometry well.
- This suggests that the latent space retains multi-primitive geometric structure for moderately complex families.

#### `Line -> Line -> Line -> Line -> Ext`

Key parameter recoverability:

- `ext1_plane_gamma/theta`: `R² = 0.974`
- `ext1_plane_phi`: `R² = 0.962`
- `line3_end_y`: `R² = 0.958`
- `line2_end_x`: `R² = 0.932`
- `line1_end_x`: `R² = 0.931`
- `line2_end_y`: `R² = 0.927`
- `ext1_sketch_size`: `R² = 0.825`

Dominant latent directions:

- `PC1` strongly tracks contour size / corner coordinates
- `PC3` strongly tracks sketch-plane orientation (`plane_gamma/theta ≈ 0.888`, `plane_phi ≈ -0.847`)

Interpretation:

- The latent structure is not limited to cylindrical families.
- Rectangular / prismatic families also show strong geometric organization in latent space.

### Practical takeaway from latent analysis

- The main bottleneck is unlikely to be a fundamentally uninterpretable DeepCAD latent space.
- Instead, the likely bottleneck is that the current image/text front-end does not recover the most important geometric control factors (e.g. radius, corner coordinates, sketch size, plane orientation) with sufficient precision.
- This supports future work on improving the visual encoder / fusion / supervision strategy rather than replacing the pretrained DeepCAD decoder.

### Run C direct vs Run B direct

- `cmd_token_acc`: `0.8081 -> 0.8319`
- `token_exact_acc`: `0.5161 -> 0.5327`
- `sequence_cmd_exact_rate`: `0.6193 -> 0.6615`
- `sequence_exact_rate`: `0.1305 -> 0.1411`
- `mean_len_abs_error`: `2.2214 -> 1.7845`

### Run C blend alpha=0.5 vs Run B blend alpha=0.5

- `cmd_token_acc`: `0.8258 -> 0.8349`
- `token_exact_acc`: `0.5495 -> 0.5570`
- `sequence_cmd_exact_rate`: `0.6854 -> 0.6987`
- `sequence_exact_rate`: `0.1683 -> 0.1760`
- `mean_len_abs_error`: `2.1432 -> 1.8822`

## Important Conclusions

1. The rescue route works.
   - Multi-view image conditioning + frozen DeepCAD latent space can produce valid CAD.

2. Direct regression already works, but retrieval-style refinement consistently improves sequence metrics.

3. `nearest` is the strongest upper-bound style enhancement.
   - It gives the best sequence-level numbers, but is closer to retrieval than pure generation.

4. `blend(alpha=0.5)` is the most balanced enhanced variant.
   - It is close to `nearest`, but easier to justify as generation + manifold refinement.

5. Expanding the train set from `117,497` to `318,886` helps the actual target we care about.
   - `pred_solid_valid_rate` improved from `0.7632` to `0.7913`.
   - `sequence_exact_rate` for direct improved from `0.1232` to `0.1305`.

6. Sequence length remains the main failure factor.
   - Short sequences are much easier.
   - Long sequences (`len_25_plus`) remain the main bottleneck.

7. The text modality already gives a measurable gain.
   - Even with a very small multimodal head and frozen BERT embeddings, Run C beats the best image-only Run B on both `direct` and `blend(alpha=0.5)` sequence metrics.

8. The current multimodal best point happens very early.
   - The best checkpoint appears at epoch `2`, before image-branch unfreezing has much time to drift.
   - This supports the interpretation that text works best as a residual correction on top of a strong frozen image baseline.

9. Freezing the image branch for the full dual-modal run is better than partial unfreezing.
   - Run D direct is the current best pure-generation dual-modal result.
   - Run D blend(alpha=0.5) is the current best balanced enhanced result.

10. Replacing GRU view fusion with a small view-token transformer substantially improves the image front-end.
   - Run E direct improves over Run B direct on command accuracy, validity, CD median and length error.
   - This supports the hypothesis that multi-view fusion, not only single-view encoding, was a major bottleneck.

11. Adding text on top of the stronger transformer image branch further improves both direct and blend results.
   - Run F direct is the current best pure-generation model.
   - Run F blend(alpha=0.5) is the current best enhanced system.

12. The DeepCAD latent/decoder ceiling is not the current bottleneck.
   - On the ARE-reference subset, `CAD_z_true` vs `CAD_true` yields `acc_cmd = 99.67%`, `acc_param = 96.49%`, `cd_median = 6.26e-4`.
   - Comparing `CAD_pred` against `CAD_z_true` instead of `CAD_true` changes the metrics only marginally.
   - Therefore the dominant error source lies in the image/text-to-latent mapping, not in the pretrained DeepCAD reconstruction ceiling.

13. The ARE-reference subset is suitable for a fairer reference comparison, but it is still not identical to ARE's simple rendered dataset.
   - It matches the DeepCAD official test source more closely by keeping only original-version test objects.
   - It does not reproduce ARE's exact 10-view 128x128 grayscale rendering protocol.

14. Focused latent-family analysis shows that several key CAD parameters are highly recoverable from DeepCAD latent vectors.
   - For example, `circle radius`, `sketch size`, `plane orientation`, and `line endpoint coordinates` achieve very high linear recoverability in simple families.
   - This supports the interpretation that the current performance gap is mainly due to front-end regression precision, not to a deficient latent geometry.

## Presentation Recommendation

- Baseline method: `image-only transformer direct`
- Main dual-modal method: `image + text transformer direct (frozen image)`
- Main improved enhanced method: `image + text transformer blend(alpha=0.5) (frozen image)`
- Analysis / upper-bound method: `nearest`
- Error-decomposition analysis: compare `CAD_pred` against `CAD_z_true`, and report `CAD_z_true` vs `CAD_true` as AE ceiling

This is the recommended framing for the midterm presentation.

## Final V1 Freeze

The code and writing should now treat the following as the fixed `v1` checkpoints:

- Pure visual checkpoint:
  - `runs/deepcad_latent/resnet18_transformer_v1_ddp2/best.pt`
- Main multimodal checkpoint:
  - `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2/best.pt`
- Final enhanced system:
  - `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2` with `blend(alpha=0.5)`

Additional note:

- The later continuation run `resnet18_transformer_text_v2_ddp2` remained stable and nearly matched `v1`, but it did not surpass `v1` on `test_mse`.
- Therefore `transformer_text_v1` remains the official multimodal best checkpoint for thesis writing and final reporting.
