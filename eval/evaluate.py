"""
评估脚本 - 实现模型评估和结果分析
"""

import torch
from tqdm import tqdm

from eval.metrics import CADMetrics


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.metrics_calculator = CADMetrics()

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches=None):
        """在数据集上评估模型"""
        self.model.eval()

        total_metrics = self.metrics_calculator.empty_metrics()
        total_batches = len(dataloader)
        effective_batches = min(total_batches, max_batches) if max_batches else total_batches
        processed_batches = 0

        for batch in tqdm(dataloader, desc='Evaluating', total=effective_batches):
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            cad_seq = batch['cad_seq'].to(self.device)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device)

            seq_len = cad_seq.shape[1]
            cmd_pred, param_pred = self.model.generate(
                images,
                text_input_ids,
                text_attention_mask,
                max_steps=seq_len
            )
            cmd_pred, param_pred = self._pad_generated_outputs(cmd_pred, param_pred, seq_len)

            gt_cmds = cad_seq[:, :, 0].long().clamp(min=0)
            gt_params = cad_seq[:, :, 1:].long()
            valid_mask = cad_valid_mask.bool()

            batch_metrics = self.metrics_calculator.compute_all_metrics(
                cmd_pred,
                param_pred.long(),
                gt_cmds,
                gt_params,
                valid_mask,
            )
            for name, value in batch_metrics.items():
                total_metrics[name] += value

            processed_batches += 1
            if max_batches and processed_batches >= max_batches:
                break

        if processed_batches == 0:
            return self.metrics_calculator.empty_metrics()

        return {name: value / processed_batches for name, value in total_metrics.items()}

    @torch.no_grad()
    def generate_and_save(self, dataloader, save_path=None, max_batches=None):
        """生成 CAD 序列并保存"""
        self.model.eval()
        generated_sequences = []
        total_batches = len(dataloader)
        effective_batches = min(total_batches, max_batches) if max_batches else total_batches
        processed_batches = 0

        for batch in tqdm(dataloader, desc='Generating', total=effective_batches):
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            sample_ids = batch['sample_ids']

            cmd_pred, param_pred = self.model.generate(
                images,
                text_input_ids,
                text_attention_mask
            )

            for idx, sample_id in enumerate(sample_ids):
                cmd_seq = cmd_pred[idx].detach().cpu()
                param_seq = param_pred[idx].detach().cpu()
                commands = [
                    {
                        'cmd': int(cmd_seq[step].item()),
                        'params': param_seq[step].tolist(),
                    }
                    for step in range(cmd_seq.shape[0])
                ]
                generated_sequences.append({
                    'sample_id': sample_id,
                    'commands': commands,
                })

            processed_batches += 1
            if max_batches and processed_batches >= max_batches:
                break

        if save_path:
            torch.save(generated_sequences, save_path)

        return generated_sequences

    def _pad_generated_outputs(self, cmd_pred, param_pred, target_len):
        """将生成序列对齐到目标长度，便于和 GT 对比。"""
        batch_size = cmd_pred.shape[0]
        current_len = cmd_pred.shape[1]
        if current_len == target_len:
            return cmd_pred, param_pred

        if current_len > target_len:
            return cmd_pred[:, :target_len], param_pred[:, :target_len, :]

        pad_len = target_len - current_len
        cmd_pad = torch.full(
            (batch_size, pad_len),
            3,
            dtype=cmd_pred.dtype,
            device=cmd_pred.device
        )
        param_pad = torch.zeros(
            batch_size,
            pad_len,
            param_pred.shape[-1],
            dtype=param_pred.dtype,
            device=param_pred.device
        )
        return torch.cat([cmd_pred, cmd_pad], dim=1), torch.cat([param_pred, param_pad], dim=1)
