# Data layout

RelScope was evaluated on public medical-imaging datasets. This repository contains derived relation metadata and patient-level splits, not the original images or label volumes.

## ToothFairy3

For tooth–IAN and tooth–sinus, pass the extracted ToothFairy3 root containing `imagesTr/` and `labelsTr/` as `--dataset-root`. The released relation paths are relative to that directory.

## TotalSegmentator v1

Pass the directory containing case folders such as `s0001/`, with `ct.nii.gz` and `segmentations/` inside each case folder.

## AMOS22

Pass the extracted AMOS22 root containing `imagesTr/` and `labelsTr/`. The pancreas and aorta masks are read from the corresponding multi-label volume.

## Medical Segmentation Decathlon Task08 Hepatic Vessel

Pass the extracted Task08 root containing `imagesTr/` and `labelsTr/`.

## Relation metadata

Each released relation record contains a public-dataset sample identifier, patient identifier, structure pair, reference physical distance, critical-region metadata, voxel spacing, and dataset-relative image/label paths. Machine-specific absolute paths and preprocessing caches have been removed.

The patient split files identify train, validation, calibration, and test samples. The scanner-subset analysis and tooth–sinus repeated out-of-fold evaluation have separate released split definitions.
