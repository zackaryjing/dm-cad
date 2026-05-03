from __future__ import annotations

from pathlib import Path

import torch

from .adapter import DeepCADAdapter
from .model import MultiModalLatentRegressor, MultiViewLatentRegressor
from .retrieval import LatentRetriever


class ImageToCadPipeline:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        backbone: str = "resnet18",
        n_views: int = 8,
        freeze_backbone: bool = False,
        retrieval_latent_root: str | Path | None = None,
        retrieval_metric: str = "cosine",
    ):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        self.is_multimodal = any(key.startswith("image_model.") or key.startswith("fusion.") for key in state_dict)

        if self.is_multimodal:
            self.model = MultiModalLatentRegressor(
                backbone_name=backbone,
                n_views=n_views,
                freeze_backbone=freeze_backbone,
            ).to(self.device)
        else:
            self.model = MultiViewLatentRegressor(
                backbone_name=backbone,
                n_views=n_views,
                freeze_backbone=freeze_backbone,
            ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.adapter = DeepCADAdapter(device=device)
        self.retriever = None
        if retrieval_latent_root is not None:
            self.retriever = LatentRetriever(
                latent_root=retrieval_latent_root,
                metric=retrieval_metric,
                device="cpu",
            )

    @torch.no_grad()
    def predict_latent(self, images: torch.Tensor, text_emb: torch.Tensor | None = None) -> torch.Tensor:
        images = images.to(self.device)
        if self.is_multimodal:
            if text_emb is None:
                raise ValueError("This checkpoint requires text embeddings but none were provided.")
            return self.model(images, text_emb.to(self.device)).cpu()
        return self.model(images).cpu()

    @torch.no_grad()
    def resolve_latent(
        self,
        pred_z: torch.Tensor,
        mode: str = "direct",
        topk: int = 1,
        blend_alpha: float = 0.5,
    ) -> dict[str, object]:
        if mode not in {"direct", "nearest", "blend"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        if mode == "direct":
            return {"final_z": pred_z.cpu(), "retrieval": None}

        if self.retriever is None:
            raise ValueError("Retrieval mode requested but no retrieval index was configured.")

        query = pred_z.cpu()
        retrieval = self.retriever.query(query, topk=topk)
        nearest_z = retrieval["latents"][:, 0, :]

        if mode == "nearest":
            final_z = nearest_z
        else:
            final_z = blend_alpha * query + (1.0 - blend_alpha) * nearest_z

        retrieval["mode"] = mode
        retrieval["blend_alpha"] = float(blend_alpha)
        return {"final_z": final_z, "retrieval": retrieval}

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor):
        return self.adapter.decode(z)
