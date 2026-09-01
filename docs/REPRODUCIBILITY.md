# Reproducibility guide

## Experiment identity

The manuscript-facing method is RelScope. The source class used by the frozen experiments is `PublicationRelMeasure3D`; retaining that name preserves compatibility with the original checkpoints and command records. Each anatomical relation is trained as a separate model.

## End-to-end order

1. Download each public dataset from its original provider.
2. Build or use the released relation index for the target anatomical pair.
3. Run `scripts/preprocess_relations.py` with the dataset root to generate the NPZ relation cache.
4. Use the released patient split. No patient appears in more than one partition within a setting.
5. Train Direct, RelScope, and the prespecified controls for seeds 20260821, 20260822, and 20260823.
6. Select checkpoints by validation MAE after the relevant warm-up period.
7. Evaluate relation-weighted MAE/P95 and patient-equal paired effects on held-out predictions.
8. Run localization, segmentation-proxy, uncertainty, distance-regime, and efficiency analyses as required.
9. Generate the unified statistics with `scripts/summarize_patterns_statistics.py`.

## Frozen optimization protocol

All functional models use AdamW with learning rate `2e-4`, weight decay `1e-4`, gradient clipping at norm 5, BF16 autocast, a maximum of 14,000 steps, validation every 400 steps, and early-stopping patience 10. RelScope uses critical/SDF weights of 0.30/0.20 and 6,000 surface-and-critical warm-up steps.

Dataset-specific crop, jitter, batch, and split settings are stored in `configs/patterns_experiments.json`.

## Segmentation comparison

Tooth–IAN includes three three-seed segmentation architectures. Additional datasets use a validation-selected pipeline: MedNeXt and the unified-protocol residual-encoder U-Net are compared for seed 20260821 at thresholds 0.3, 0.5, and 0.7. Selection minimizes distance MAE plus ten times the empty-mask failure rate, after which the selected architecture is completed for the remaining seeds.

The residual-encoder U-Net is the architecture implemented in `external_baselines.py` under the unified study protocol. It is not presented as the complete official nnU-Net recipe.

## Statistical units

Relation-weighted MAE gives each indexed anatomical relation equal weight. Patient-equal effects first average relations within a patient and then compare Direct minus RelScope, so positive values favor RelScope. Resampling and paired tests use the patient as the inferential unit.

The predicted scale is evaluated separately for case ranking and split-conformal interval construction. Marginal interval coverage is not treated as evidence of reliable failure ranking.

## Released and excluded artifacts

Released:

- source code and command-line analysis scripts;
- frozen experiment configuration;
- path-portable relation indices and patient splits;
- tooth–sinus fold definitions;
- aggregate and patient-level manuscript statistics.

Not redistributed:

- medical images and label volumes;
- preprocessed NPZ caches;
- checkpoints and training logs.

These exclusions do not change the published protocol. Images must be obtained under the original dataset terms, and caches can be rebuilt from the released relation metadata.
