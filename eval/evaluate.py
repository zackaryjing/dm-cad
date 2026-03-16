"""
评估脚本 - 实现模型评估和结果分析
"""

import torch
from tqdm import tqdm

from eval.metrics import CADMetrics


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 待评估模型
            device: 评估设备
        """
        self.model = model
        self.device = device
        self.metrics_calculator = CADMetrics()

    @torch.no_grad()
    def evaluate(self, dataloader):
        """在数据集上评估模型

        Args:
            dataloader: 数据加载器
        Returns:
            results: 评估结果字典
        """
        self.model.eval()

        all_pred_cmds = []
        all_param_preds = []
        all_gt_cmds = []
        all_gt_params = []
        all_valid_masks = []

        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            cad_seq = batch['cad_seq'].to(self.device)
            cad_valid_mask = batch['cad_valid_mask'].to(self.device)

            # 模型预测
            cmd_logits, param_pred = self.model(
                images, text_input_ids, text_attention_mask
            )

            # 获取预测命令
            cmd_pred = torch.argmax(cmd_logits, dim=-1)

            # 收集结果
            all_pred_cmds.append(cmd_pred)
            all_param_preds.append(param_pred)

            # Ground truth
            gt_cmds = cad_seq[:, :, 0].long()
            gt_params = cad_seq[:, :, 1:]
            all_gt_cmds.append(gt_cmds)
            all_gt_params.append(gt_params)
            all_valid_masks.append(cad_valid_mask)

        # 合并所有批次结果
        pred_cmds = torch.cat(all_pred_cmds, dim=0)
        param_preds = torch.cat(all_param_preds, dim=0)
        gt_cmds = torch.cat(all_gt_cmds, dim=0)
        gt_params = torch.cat(all_gt_params, dim=0)
        valid_masks = torch.cat(all_valid_masks, dim=0)

        # 计算指标
        metrics = self.metrics_calculator.compute_all_metrics(
            pred_cmds, param_preds, gt_cmds, gt_params, valid_masks
        )

        return metrics

    @torch.no_grad()
    def generate_and_save(self, dataloader, save_path=None):
        """生成 CAD 序列并保存

        Args:
            dataloader: 数据加载器
            save_path: 保存路径
        Returns:
            generated_sequences: 生成的序列列表
        """
        self.model.eval()
        generated_sequences = []

        for batch in tqdm(dataloader, desc='Generating'):
            images = batch['images'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            uids = batch['uids']

            # 生成
            generated = self.model.generate(
                images, text_input_ids, text_attention_mask
            )

            # 保存结果
            for uid, gen in zip(uids, generated):
                seq_data = {'uid': uid, 'commands': gen}
                generated_sequences.append(seq_data)

        # 保存
        if save_path:
            torch.save(generated_sequences, save_path)

        return generated_sequences
