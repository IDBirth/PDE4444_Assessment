# Data Acquisition Setup

## Physical setup

The inspection system assumes a fixed visual inspection area:

- an object is placed in a repeatable capture position
- a camera captures the cup image
- the model predicts whether the cup should pass or fail

This is a standard engineering quality-control setup because it reduces
viewpoint variation and makes the classification problem tractable.

## Label definition

Two classes are used:

- `defect`  -> `FAIL`
- `non_defect` -> `PASS`

The labels are verifiable from the image, which is important for the
assessment requirement that ground truth must be clearly defined.

## Raw data locations

- [zeroq_cup_classification_scaffold/data/raw/defective](/home/ubu/Desktop/Assessment/zeroq_cup_classification_scaffold/data/raw/defective)
- [zeroq_cup_classification_scaffold/data/raw/non_defective](/home/ubu/Desktop/Assessment/zeroq_cup_classification_scaffold/data/raw/non_defective)

## Why preprocessing was necessary

The original dataset was heavily imbalanced toward defective examples.
That made early models, especially classical baselines, prone to learning
the dominant class rather than the actual visual distinction.

To fix that:

1. the majority class was undersampled using
   [balance_cleaned_dataset.py](/home/ubu/Desktop/Assessment/zeroq_cup_classification_scaffold/scripts/balance_cleaned_dataset.py)
2. the balanced set was split with stratification using
   [prepare_dataset.py](/home/ubu/Desktop/Assessment/defect_classification_stack/prepare_dataset.py)

## Final balanced dataset

- Total images: `690`
- Train: `483`
- Validation: `69`
- Test: `138`

Metadata file:

- [runs5/data_balanced/dataset_metadata.json](/home/ubu/Desktop/Assessment/runs5/data_balanced/dataset_metadata.json)

## Why this matters in the demo

If asked why the dataset setup matters, the short answer is:

> In inspection problems, a biased dataset produces biased decisions. The
> balancing and stratified split were necessary so that the model learned
> visual defect cues rather than the majority-class frequency.
