# Demo Commands

All commands assume the working directory is:

```bash
cd /home/ubu/Desktop/Assessment
```

## 1. Recommended main demo: YOLO26n-cls on live stream

Use this if the live stream is available.

```bash
.venv/bin/python defect_classification_stack/demo_live_test_with_video.py \
  --model /home/ubu/Desktop/Assessment/runs5/iter1_yolo/train/weights/best.pt \
  --stream-url "WWW.URL.COM" \
  --auth-user "YOUR_USER" \
  --auth-password "YOUR_PASSWORD"
```

Why this is the recommended command:

- best deployment model by accuracy in `runs5`
- live overlay display is already implemented
- uses the repaired `runs5` YOLO checkpoint

## 2. Backup demo: YOLO26n-cls on offline unseen input

Use this if the stream is unavailable.

```bash
.venv/bin/python defect_classification_stack/demo_live_test_with_video.py \
  --model /home/ubu/Desktop/Assessment/runs5/iter1_yolo/train/weights/best.pt \
  --input /home/ubu/Desktop/Assessment/PATH/TO/UNSEEN_IMAGES_OR_VIDEO
```

## 3. Optional comparison demo: top-model panel

This is useful if you want to show several trained models side by side.

```bash
.venv/bin/python defect_classification_stack/demo_live_test_with_video_top6.py \
  --stream-url "WWW.URL.COM" \
  --auth-user "YOUR_USER" \
  --auth-password "YOUR_PASSWORD" \
  --models top_models/01_yolo26n_320/model.pt \
           top_models/02_yolo26s_320/model.pt \
           top_models/03_mobilenet_selu_r3/model.pt \
           top_models/04_mlp_boost_r3/model.pt \
           top_models/05_mobilenet_elu_r1/model.pt
```

## 4. Results files to show immediately after the run

- [runs5/iter1_yolo/metrics.json](/home/ubu/Desktop/Assessment/runs5/iter1_yolo/metrics.json)
- [runs5/iter1_yolo/train/results.png](/home/ubu/Desktop/Assessment/runs5/iter1_yolo/train/results.png)
- [runs5/final_report/model_comparison_acc.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_acc.png)
- [runs5/final_report/model_comparison_f1.png](/home/ubu/Desktop/Assessment/runs5/final_report/model_comparison_f1.png)

## 5. Existing working command reference

If you want the exact stream commands previously used during development,
refer to:

- [run.txt](/home/ubu/Desktop/Assessment/run.txt)
