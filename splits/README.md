# Released splits and relation indices

Each setting directory contains a patient-level `split.json` and, where applicable, a `relation_index.json`. Paths in relation records are relative to the original dataset root and are resolved by `scripts/preprocess_relations.py --dataset-root ...`.

The tooth–sinus directory also contains the five repeated-evaluation folds and their aggregate cross-validation summary. The scanner-subset split is provided separately under `tooth_ian_scanner_holdout/`.

Machine-specific absolute paths and invalid-file audit entries were removed from the public copies. Sample identifiers, patient partitions, relation definitions, reference distances, and geometric metadata are preserved.
