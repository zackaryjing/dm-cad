# Final Thesis Results Check (2026-05-15)
## Official checkpoints
- Image-only: `runs/deepcad_latent/resnet18_transformer_v1_ddp2/best.pt`
- Multimodal direct: `runs/deepcad_latent/resnet18_transformer_text_v1_ddp2/best.pt`
- Multimodal enhanced: `transformer_text_v1 + blend(alpha=0.5)`
## Best training records
- Image-only transformer best `test_mse`: epoch 12 = 0.060414
- Multimodal transformer v1 best `test_mse`: epoch 10 = 0.059321
- Multimodal transformer v2 best `test_mse`: epoch 10 = 0.059358 (not adopted)
## Main results on held-out test set
| Method | Seq Exact | Solid Valid | ACC_cmd | ACC_param | Invalidity | CD mean | CD median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Image-only Transformer Direct | 0.1313 | 0.8135 | 0.8634 | 0.7415 | 0.1865 | 0.12981 | 0.03433 |
| Multimodal Transformer Direct | 0.1423 | 0.8450 | 0.8721 | 0.7528 | 0.1550 | 0.12944 | 0.02992 |
| Multimodal Transformer Blend 0.5 | 0.1742 | 0.9464 | 0.8620 | 0.7720 | 0.0536 | 0.12508 | 0.03365 |

## ARE-reference subset
| Method | Seq Exact | Solid Valid | ACC_cmd | ACC_param | Invalidity | CD mean | CD median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multimodal Transformer Direct | 0.1462 | 0.8480 | 0.8775 | 0.7577 | 0.1520 | 0.12649 | 0.02712 |
| Multimodal Transformer Blend 0.5 | 0.1795 | 0.9478 | 0.8680 | 0.7766 | 0.0522 | 0.12248 | 0.03090 |

## Consistency notes
- The historical snapshot dated 2026-04-16 is retained for chronology, but later transformer fine-tuning improved the image-only transformer best checkpoint.
- The old report material tables were GRU-centered and should not be used as the final thesis tables.
- The official final multimodal checkpoint remains `resnet18_transformer_text_v1_ddp2/best.pt`; `v2` did not surpass it on test MSE.
