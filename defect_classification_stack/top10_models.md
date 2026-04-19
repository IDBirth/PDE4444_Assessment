# Top 10 Models

Chosen from the ranked results in `defect_classification_stack/PROGRESS_AND_RESULTS.md`, taking the strongest available checkpoint for each model variant across `runs`, `runs2`, and `runs3`.

| Rank | Model | F1 | Checkpoint Path |
|------|-------|----|-----------------|
| 1 | `yolo26n_cls_320` | `1.0000` | `/home/ubu/Desktop/Assessment/runs3/iter1_yolo_320/train/weights/best.pt` |
| 2 | `yolo26s_cls_320` | `1.0000` | `/home/ubu/Desktop/Assessment/runs3/iter2_yolo_s/train/weights/best.pt` |
| 3 | `mobilenet_selu` fine-tuned | `0.9517` | `/home/ubu/Desktop/Assessment/runs3/iter3_mobilenet_finetune/selu/model.pt` |
| 4 | `mlp_random_search` boosted | `0.9517` | `/home/ubu/Desktop/Assessment/runs3/iter1_mlp_boost/best_model.pt` |
| 5 | `MobileNetV2-ELU` fine-tuned | `0.9517` | `/home/ubu/Desktop/Assessment/defect_classification_stack/runs/iter3_mobilenet_finetune/elu/model.pt` |
| 6 | `mobilenet_gelu` fine-tuned | `0.9504` | `/home/ubu/Desktop/Assessment/runs2/iter3_mobilenet_finetune/gelu/model.pt` |
| 7 | `MobileNetV2-LeakyReLU` fine-tuned | `0.9452` | `/home/ubu/Desktop/Assessment/defect_classification_stack/runs/iter3_mobilenet_finetune/leaky_relu/model.pt` |
| 8 | `mobilenet_relu` fine-tuned | `0.9444` | `/home/ubu/Desktop/Assessment/runs3/iter3_mobilenet_finetune/relu/model.pt` |
| 9 | `MLP HParam Search (ELU, 256)` | `0.9437` | `/home/ubu/Desktop/Assessment/defect_classification_stack/runs/iter1_mlp_search/best_model.pt` |
| 10 | `MobileNetV2-GELU` fine-tuned | `0.9429` | `/home/ubu/Desktop/Assessment/defect_classification_stack/runs/iter3_mobilenet_finetune/gelu/model.pt` |

## Source Tables

- `runs1` final rankings: `defect_classification_stack/PROGRESS_AND_RESULTS.md`
- `runs2` scoreboard: `defect_classification_stack/PROGRESS_AND_RESULTS.md`
- `runs3` scoreboard: `defect_classification_stack/PROGRESS_AND_RESULTS.md`
