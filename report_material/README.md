# Report Material Index

本目录用于存放项目设计报告可直接复用的图表、表格和摘要。

## 目录结构

- `figures/`
  - `old_framework_training_curves.png`：旧框架训练/验证关键曲线
  - `new_route_training_curves.png`：新框架三组训练策略的测试曲线
  - `main_method_comparison.png`：主方法横向对比图
  - `bucket_solid_validity.png`：不同序列长度分桶下的实体有效率
- `tables/`
  - `dataset_summary.md`
  - `method_comparison.md`
  - `bucket_solid_validity.md`
  - `old_framework_best_scalars.md`
- `data/`
  - `old_framework_scalars.json`
  - `old_framework_best_scalars.json`
  - `new_route_histories.json`
  - `new_route_eval_summaries.json`

## 可直接写进报告的关键结论

1. 旧框架在 teacher-forced 训练曲线上出现了看似乐观的指标，但自由生成阶段稳定性不足，促使框架切换。
2. 新框架在冻结 DeepCAD latent 空间后，可以稳定输出可用 CAD，并且随着训练集从 `117,497` 扩大到 `318,886`，实体有效率进一步提升。
3. 双模态确实有效：
   - `Image-only Direct` 的实体有效率为 `0.7913`
   - `Image+Text Direct (Frozen)` 提升到 `0.8308`
4. 当前最好结果为 `Image+Text Blend 0.5 (Frozen)`：
   - `Token Exact Acc = 0.5640`
   - `Sequence Exact Rate = 0.1786`
   - `Solid Valid Rate = 0.9456`
5. 长序列仍然是主要难点，但双模态和 blend 在中短序列以及中等长度样本上均显著改善。

## 备注

- 生成脚本：`report_material/generate_materials.py`
- 若后续有新实验结果，可重新运行该脚本覆盖更新图表。
