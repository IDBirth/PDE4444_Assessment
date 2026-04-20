# Live Demo Prep

This folder packages the live demonstration material for the PDE4444
assessment so the demo can be delivered consistently under time pressure.

## What is here

- `01_demo_talk_track.md`
  2-3 minute explanation of the engineering problem, system, dataset,
  modelling choices, and headline results.
- `02_demo_runbook.md`
  Step-by-step live demo flow: what to show, what to say, which commands
  to run, and what fallback path to use if the stream is unavailable.
- `03_data_acquisition_setup.md`
  Short explanation of how the image data was collected, cleaned,
  balanced, and split for training.
- `04_technical_qa.md`
  Direct answers for likely viva/demo questions on activation functions,
  optimisation methods, and model limitations.
- `05_demo_commands.md`
  Ready-to-run commands for the best single-model demo and the optional
  top-model comparison demo.
- `06_demo_evidence_index.md`
  File map of the exact artefacts to open during the demo or refer to
  during questioning.

## Recommended demo path

Use the `YOLO26n-cls` model as the main live demo model:

- It is the strongest model by test accuracy in `runs5`
- It produced `top1 = 0.9928` on the balanced test set
- It is fast and easy to explain in a live setting

Primary evidence:

- [runs5/final_report/all_results.csv](/home/ubu/Desktop/Assessment/runs5/final_report/all_results.csv)
- [runs5/final_report/model_comparison_f1.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_f1.png)
- [runs5/final_report/model_comparison_acc.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_acc.png)
- [runs5/iter1_yolo/metrics.json](/home/ubu/Desktop/Assessment/runs5/iter1_yolo/metrics.json)
- [runs5/iter1_yolo/train/results.png](/home/ubu/Desktop/Assessment/runs5/iter1_yolo/train/results.png)

## Recommended opening sequence

1. Open `01_demo_talk_track.md`
2. Keep `02_demo_runbook.md` visible during the demo
3. Run the command from `05_demo_commands.md`
4. If questioned, answer from `04_technical_qa.md`
