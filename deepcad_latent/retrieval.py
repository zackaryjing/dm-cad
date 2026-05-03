from __future__ import annotations

from pathlib import Path

import torch

from .data import load_latent_shards


class LatentRetriever:
    def __init__(
        self,
        latent_root: str | Path,
        metric: str = "cosine",
        device: str | torch.device = "cpu",
    ):
        if metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported retrieval metric: {metric}")

        latent_by_id = load_latent_shards(latent_root)
        self.sample_ids = list(latent_by_id.keys())
        self.latents = torch.stack([latent_by_id[sample_id] for sample_id in self.sample_ids], dim=0)
        self.metric = metric
        self.device = torch.device(device)
        self.latents = self.latents.to(self.device, dtype=torch.float32)
        self._latents_norm = None
        if self.metric == "cosine":
            self._latents_norm = torch.nn.functional.normalize(self.latents, dim=-1)

    @torch.no_grad()
    def query(self, query_z: torch.Tensor, topk: int = 1) -> dict[str, object]:
        if topk <= 0:
            raise ValueError("topk must be positive")

        query_z = query_z.to(self.device, dtype=torch.float32)
        if query_z.ndim == 1:
            query_z = query_z.unsqueeze(0)

        if self.metric == "cosine":
            query_norm = torch.nn.functional.normalize(query_z, dim=-1)
            scores = query_norm @ self._latents_norm.t()
            values, indices = torch.topk(scores, k=min(topk, scores.shape[1]), dim=-1, largest=True)
        else:
            distances = torch.cdist(query_z, self.latents, p=2)
            values, indices = torch.topk(distances, k=min(topk, distances.shape[1]), dim=-1, largest=False)

        retrieved_latents = self.latents[indices]
        retrieved_ids = [[self.sample_ids[int(i)] for i in row] for row in indices.cpu()]
        return {
            "indices": indices.cpu(),
            "scores": values.cpu(),
            "sample_ids": retrieved_ids,
            "latents": retrieved_latents.detach().cpu(),
        }
