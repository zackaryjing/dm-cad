"""
Dual-Modal CAD Generator (DM-CAD)

基于多视图图像与文本描述的参数化 CAD 序列生成网络

Author: AI Assistant
Date: 2026-03-13
"""

from .models import (
    ViewEncoder,
    MultiViewFusion,
    TextEncoder,
    ModalFusion,
    CADDecoder,
    DualModalCADGenerator
)
from .data import (
    CADDataset,
    CADRenderer,
    CADDataAugmentation
)
from .train import (
    CADLoss,
    Trainer
)
from .eval import (
    Evaluator,
    evaluate_sequence_accuracy,
    evaluate_parameter_accuracy
)

__version__ = '1.0.0'
__all__ = [
    # Models
    'ViewEncoder',
    'MultiViewFusion',
    'TextEncoder',
    'ModalFusion',
    'CADDecoder',
    'DualModalCADGenerator',
    # Data
    'CADDataset',
    'CADRenderer',
    'CADDataAugmentation',
    # Train
    'CADLoss',
    'Trainer',
    # Eval
    'Evaluator',
    'evaluate_sequence_accuracy',
    'evaluate_parameter_accuracy',
]
