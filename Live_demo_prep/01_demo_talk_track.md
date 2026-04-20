# 2-3 Minute Demo Talk Track

## Short version

My engineering problem is automated visual quality inspection for cups:
given a camera image, the system must decide whether the object should
pass or fail quality control. In engineering terms, the input is an image
captured from a fixed inspection area, and the output is a binary
decision: `PASS` for non-defective and `FAIL` for defective.

I chose this problem because it is a realistic industrial inspection task.
Manual inspection is slow, inconsistent, and difficult to scale. A machine
learning system can provide faster and more repeatable decisions, which is
important when defective products must be rejected reliably.

The dataset originally had a severe class imbalance. The raw set contained
far more defective examples than non-defective ones, and that caused early
classical baselines to collapse toward one-class behaviour. To correct
that, I balanced the dataset and built a reproducible train/validation/test
split. After balancing, the final working dataset used 690 images total,
with 483 for training, 69 for validation, and 138 for test.

I compared several model families:

- classical baselines using HOG features with SVM and sklearn MLP
- a scratch CNN with different activation functions
- an optimiser comparison experiment using the same small MLP
- a PyTorch MLP random search
- transfer learning with MobileNetV2
- a pretrained YOLO26n classifier

The strongest model was `YOLO26n-cls`, because it combines a pretrained
feature extractor with a lightweight classification pipeline. In `runs5`,
it achieved `99.28%` top-1 accuracy on the balanced test set. That made it
the best model by accuracy, while the best F1 among the non-YOLO models
came from the fine-tuned MobileNetV2-ReLU and the MLP random search, both
at `0.9545` F1.

For the live demo, I will show the data acquisition setup, run the trained
model on unseen stream frames or unseen images, display prediction outputs,
and then explain why the model predicts `PASS` or `FAIL` based on the
learned visual patterns. I will also explain the trade-offs between
classical models, scratch CNNs, transfer learning, and YOLO.

## Key numbers to memorise

- Problem: binary cup inspection, `PASS` vs `FAIL`
- Final balanced dataset: `690` images
- Train / val / test: `483 / 69 / 138`
- Best classical baseline: `hog_svm`, F1 `0.8346`
- Best MLP random search in `runs5`: F1 `0.9545`
- Best overall by accuracy in `runs5`: `yolo26n-cls.pt`, top-1 `0.9928`

## One-line justification for model choice

I chose YOLO26n-cls for the live demo because it gave the highest test
accuracy, runs fast enough for live video, and is easier to deploy for
engineering inspection than a larger or slower model.
