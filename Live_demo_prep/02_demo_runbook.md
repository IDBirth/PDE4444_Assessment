# Live Demo Runbook

## Goal

Show a complete engineering ML workflow:

1. define the problem
2. show how data was acquired and prepared
3. run a trained model on unseen data
4. display predictions live
5. explain behaviour, strengths, and limitations

## Before the demo

Open these files in advance:

- [runs5/final_report/model_comparison_acc.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_acc.png)
- [runs5/final_report/model_comparison_f1.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_f1.png)
- [runs5/iter1_yolo/train/results.png](/home/ubu/Desktop/Assessment/runs5/iter1_yolo/train/results.png)
- [runs5/iter1_mlp_search/convergence.png](/home/ubu/Desktop/Assessment/runs5/iter1_mlp_search/convergence.png)
- [runs5/iter1_optimizer/optimizer_convergence.png](/home/ubu/Desktop/Assessment/runs5/iter1_optimizer/optimizer_convergence.png)
- [runs5/iter1_crossval/overfitting_analysis.png](/home/ubu/Desktop/Assessment/runs5/iter1_crossval/overfitting_analysis.png)

Keep these notes visible:

- `01_demo_talk_track.md`
- `04_technical_qa.md`

## Suggested live flow

### 1. Explain the problem

Use the first section of `01_demo_talk_track.md`.

What to show:

- a few sample defect / non-defect images
- the problem statement: image in, pass/fail out

Useful folders:

- [zeroq_cup_classification_scaffold/data/raw/defective](/home/ubu/Desktop/Assessment/zeroq_cup_classification_scaffold/data/raw/defective)
- [zeroq_cup_classification_scaffold/data/raw/non_defective](/home/ubu/Desktop/Assessment/zeroq_cup_classification_scaffold/data/raw/non_defective)

### 2. Show the data acquisition and preparation setup

Explain:

- fixed inspection area
- camera captures image
- images labelled as defect / non-defect
- dataset balanced by undersampling the majority class
- stratified train/val/test split

Useful artefacts:

- [runs5/data_balanced/dataset_metadata.json](/home/ubu/Desktop/Assessment/runs5/data_balanced/dataset_metadata.json)
- [draw.io/dataset_processing_raw_to_balanced.drawio](/home/ubu/Desktop/Assessment/draw.io/dataset_processing_raw_to_balanced.drawio)

### 3. Show final model selection

Say:

- I compared classical baselines, scratch CNNs, optimisers, MLP search,
  transfer learning, and YOLO
- the best deployment model by accuracy was YOLO26n-cls
- the best non-YOLO neural models in `runs5` were MobileNetV2 fine-tuned
  and MLP random search

Show:

- [runs5/final_report/model_comparison_acc.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_acc.png)
- [runs5/final_report/model_comparison_f1.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_f1.png)

### 4. Run the live prediction demo

Recommended primary demo:

- single-model YOLO26n live run using
  `defect_classification_stack/demo_live_test_with_video.py`

Recommended backup demo:

- offline input path using the same script
- top-model comparison using
  `defect_classification_stack/demo_live_test_with_video_top6.py`

Use the commands in `05_demo_commands.md`.

### 5. Explain prediction outputs

During the live demo, explain:

- the model outputs a class label and confidence
- `FAIL` means the frame resembles the defective class
- `PASS` means the frame resembles the non-defective class
- confidence reflects the strength of the class preference, not absolute
  certainty

### 6. Answer technical questions

Use `04_technical_qa.md`.

## Fallback strategy

If the stream fails:

1. say the live inference pipeline is identical for video and offline input
2. switch to the offline command in `05_demo_commands.md`
3. show prediction outputs on unseen images or a saved video clip

If YOLO demo behaves unexpectedly:

1. show `runs5/iter1_yolo/train/results.png`
2. show `runs5/iter1_yolo/metrics.json`
3. explain that the same trained checkpoint is being used in the live demo
