from __future__ import annotations

import io
import math
import pickle
from pathlib import Path

import h5py
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_LMDB_ENV_CACHE: dict[str, lmdb.Environment] = {}


def load_ids(ids_file: str | Path) -> list[str]:
    path = Path(ids_file)
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def load_latent_shards(latent_root: str | Path) -> dict[str, torch.Tensor]:
    latent_root = Path(latent_root)
    shard_paths = sorted(latent_root.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No latent shards found under: {latent_root}")

    latent_by_id: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu")
        sample_ids = payload["sample_ids"]
        zs = payload["z"].to(torch.float32)
        for sample_id, z in zip(sample_ids, zs):
            latent_by_id[sample_id] = z.clone()
    return latent_by_id


def load_embedding_shards(root: str | Path, tensor_key: str) -> dict[str, torch.Tensor]:
    root = Path(root)
    shard_paths = sorted(root.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found under: {root}")

    data_by_id: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        payload = torch.load(shard_path, map_location="cpu")
        sample_ids = payload["sample_ids"]
        vectors = payload[tensor_key].to(torch.float32)
        for sample_id, vector in zip(sample_ids, vectors):
            data_by_id[sample_id] = vector.clone()
    return data_by_id


def build_image_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_raw_cad_vec(data_root: str | Path, sample_id: str) -> np.ndarray:
    data_root = Path(data_root)
    group_id, sample_name = sample_id.split("/")[:2]
    path = data_root / "cad_vec" / group_id / f"{sample_name}.h5"
    with h5py.File(path, "r") as f:
        key = next(iter(f.keys()))
        cad_vec = f[key][:]
    if cad_vec.ndim == 1:
        cad_vec = cad_vec.reshape(-1, 17)
    return cad_vec.astype(np.int64, copy=False)


class ImageLatentDataset(Dataset):
    def __init__(
        self,
        ids_file: str | Path,
        latent_root: str | Path,
        data_root: str | Path,
        lmdb_path: str | Path = "cad_data.lmdb",
        img_size: int = 224,
        n_views: int = 8,
    ):
        self.sample_ids = load_ids(ids_file)
        self.latent_by_id = load_latent_shards(latent_root)
        self.sample_ids = [sample_id for sample_id in self.sample_ids if sample_id in self.latent_by_id]
        if not self.sample_ids:
            raise ValueError("No IDs matched the latent shards.")

        self.data_root = Path(data_root)
        self.lmdb_path = self.data_root / lmdb_path if not Path(lmdb_path).is_absolute() else Path(lmdb_path)
        self.n_views = n_views
        self.image_transform = build_image_transform(img_size)

        self._env = None
        self._txn = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def _ensure_lmdb_open(self):
        if self._env is not None and self._txn is not None:
            return
        env_key = str(self.lmdb_path.resolve())
        if env_key not in _LMDB_ENV_CACHE:
            _LMDB_ENV_CACHE[env_key] = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=self.lmdb_path.is_dir(),
            )
        self._env = _LMDB_ENV_CACHE[env_key]
        self._txn = self._env.begin(write=False)

    def _load_images(self, sample_id: str) -> torch.Tensor:
        self._ensure_lmdb_open()
        payload = self._txn.get(sample_id.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Sample not found in LMDB: {sample_id}")

        record = pickle.loads(payload)
        image_bytes_list = record.get("image_bytes", [])
        images = []
        for image_bytes in image_bytes_list[: self.n_views]:
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.image_transform.transforms[0].size[0], self.image_transform.transforms[0].size[1])
            images.append(img)

        while len(images) < self.n_views:
            images.append(torch.ones_like(images[0]))
        return torch.stack(images)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        return {
            "sample_id": sample_id,
            "images": self._load_images(sample_id),
            "z": self.latent_by_id[sample_id],
        }


class ImageLatentCadDataset(Dataset):
    def __init__(
        self,
        ids_file: str | Path,
        latent_root: str | Path,
        data_root: str | Path,
        max_total_len: int,
        eos_vec: np.ndarray,
        lmdb_path: str | Path = "cad_data.lmdb",
        img_size: int = 224,
        n_views: int = 8,
    ):
        self.sample_ids = load_ids(ids_file)
        self.latent_by_id = load_latent_shards(latent_root)
        self.sample_ids = [sample_id for sample_id in self.sample_ids if sample_id in self.latent_by_id]
        if not self.sample_ids:
            raise ValueError("No IDs matched the latent shards.")

        self.data_root = Path(data_root)
        self.lmdb_path = self.data_root / lmdb_path if not Path(lmdb_path).is_absolute() else Path(lmdb_path)
        self.n_views = n_views
        self.max_total_len = int(max_total_len)
        self.eos_vec = np.asarray(eos_vec, dtype=np.int64)
        self.image_transform = build_image_transform(img_size)

        self._env = None
        self._txn = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def _ensure_lmdb_open(self):
        if self._env is not None and self._txn is not None:
            return
        env_key = str(self.lmdb_path.resolve())
        if env_key not in _LMDB_ENV_CACHE:
            _LMDB_ENV_CACHE[env_key] = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=self.lmdb_path.is_dir(),
            )
        self._env = _LMDB_ENV_CACHE[env_key]
        self._txn = self._env.begin(write=False)

    def _load_images(self, sample_id: str) -> torch.Tensor:
        self._ensure_lmdb_open()
        payload = self._txn.get(sample_id.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Sample not found in LMDB: {sample_id}")

        record = pickle.loads(payload)
        image_bytes_list = record.get("image_bytes", [])
        images = []
        for image_bytes in image_bytes_list[: self.n_views]:
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.image_transform.transforms[0].size[0], self.image_transform.transforms[0].size[1])
            images.append(img)

        while len(images) < self.n_views:
            images.append(torch.ones_like(images[0]))
        return torch.stack(images)

    def _pad_cad_vec(self, cad_vec: np.ndarray) -> np.ndarray:
        array = np.asarray(cad_vec, dtype=np.int64)
        if array.ndim == 1:
            array = array.reshape(-1, 17)
        if array.shape[0] > self.max_total_len:
            raise ValueError(
                f"cad_vec length {array.shape[0]} exceeds max_total_len={self.max_total_len} for sample"
            )
        if array.shape[0] == self.max_total_len:
            return array
        pad_len = self.max_total_len - array.shape[0]
        padding = np.repeat(self.eos_vec[None, :], pad_len, axis=0)
        return np.concatenate([array, padding], axis=0)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        cad_vec = load_raw_cad_vec(self.data_root, sample_id)
        cad_vec = self._pad_cad_vec(cad_vec)
        return {
            "sample_id": sample_id,
            "images": self._load_images(sample_id),
            "z": self.latent_by_id[sample_id],
            "cad_vec": torch.as_tensor(cad_vec, dtype=torch.long),
        }


class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        ids_file: str | Path,
        data_root: str | Path,
        lmdb_path: str | Path = "cad_data.lmdb",
        img_size: int = 224,
        n_views: int = 8,
    ):
        self.sample_ids = load_ids(ids_file)
        self.data_root = Path(data_root)
        self.lmdb_path = self.data_root / lmdb_path if not Path(lmdb_path).is_absolute() else Path(lmdb_path)
        self.n_views = n_views
        self.image_transform = build_image_transform(img_size)
        self._env = None
        self._txn = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def _ensure_lmdb_open(self):
        if self._env is not None and self._txn is not None:
            return
        env_key = str(self.lmdb_path.resolve())
        if env_key not in _LMDB_ENV_CACHE:
            _LMDB_ENV_CACHE[env_key] = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=self.lmdb_path.is_dir(),
            )
        self._env = _LMDB_ENV_CACHE[env_key]
        self._txn = self._env.begin(write=False)

    def _load_images(self, sample_id: str) -> torch.Tensor:
        self._ensure_lmdb_open()
        payload = self._txn.get(sample_id.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Sample not found in LMDB: {sample_id}")

        record = pickle.loads(payload)
        image_bytes_list = record.get("image_bytes", [])
        images = []
        for image_bytes in image_bytes_list[: self.n_views]:
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.image_transform.transforms[0].size[0], self.image_transform.transforms[0].size[1])
            images.append(img)

        while len(images) < self.n_views:
            images.append(torch.ones_like(images[0]))
        return torch.stack(images)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        return {
            "sample_id": sample_id,
            "images": self._load_images(sample_id),
        }


class ImageTextLatentDataset(Dataset):
    def __init__(
        self,
        ids_file: str | Path,
        latent_root: str | Path,
        text_root: str | Path,
        data_root: str | Path,
        lmdb_path: str | Path = "cad_data.lmdb",
        img_size: int = 224,
        n_views: int = 8,
    ):
        self.sample_ids = load_ids(ids_file)
        self.latent_by_id = load_latent_shards(latent_root)
        self.text_by_id = load_embedding_shards(text_root, tensor_key="text_emb")
        self.sample_ids = [
            sample_id
            for sample_id in self.sample_ids
            if sample_id in self.latent_by_id and sample_id in self.text_by_id
        ]
        if not self.sample_ids:
            raise ValueError("No IDs matched both latent and text embedding shards.")

        self.data_root = Path(data_root)
        self.lmdb_path = self.data_root / lmdb_path if not Path(lmdb_path).is_absolute() else Path(lmdb_path)
        self.n_views = n_views
        self.image_transform = build_image_transform(img_size)
        self._env = None
        self._txn = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def _ensure_lmdb_open(self):
        if self._env is not None and self._txn is not None:
            return
        env_key = str(self.lmdb_path.resolve())
        if env_key not in _LMDB_ENV_CACHE:
            _LMDB_ENV_CACHE[env_key] = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=self.lmdb_path.is_dir(),
            )
        self._env = _LMDB_ENV_CACHE[env_key]
        self._txn = self._env.begin(write=False)

    def _load_images(self, sample_id: str) -> torch.Tensor:
        self._ensure_lmdb_open()
        payload = self._txn.get(sample_id.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Sample not found in LMDB: {sample_id}")

        record = pickle.loads(payload)
        image_bytes_list = record.get("image_bytes", [])
        images = []
        for image_bytes in image_bytes_list[: self.n_views]:
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.image_transform.transforms[0].size[0], self.image_transform.transforms[0].size[1])
            images.append(img)

        while len(images) < self.n_views:
            images.append(torch.ones_like(images[0]))
        return torch.stack(images)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        return {
            "sample_id": sample_id,
            "images": self._load_images(sample_id),
            "text_emb": self.text_by_id[sample_id],
            "z": self.latent_by_id[sample_id],
        }


class ImageTextOnlyDataset(Dataset):
    def __init__(
        self,
        ids_file: str | Path,
        text_root: str | Path,
        data_root: str | Path,
        lmdb_path: str | Path = "cad_data.lmdb",
        img_size: int = 224,
        n_views: int = 8,
    ):
        self.sample_ids = load_ids(ids_file)
        self.text_by_id = load_embedding_shards(text_root, tensor_key="text_emb")
        self.sample_ids = [sample_id for sample_id in self.sample_ids if sample_id in self.text_by_id]
        if not self.sample_ids:
            raise ValueError("No IDs matched the text embedding shards.")

        self.data_root = Path(data_root)
        self.lmdb_path = self.data_root / lmdb_path if not Path(lmdb_path).is_absolute() else Path(lmdb_path)
        self.n_views = n_views
        self.image_transform = build_image_transform(img_size)
        self._env = None
        self._txn = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        return state

    def _ensure_lmdb_open(self):
        if self._env is not None and self._txn is not None:
            return
        env_key = str(self.lmdb_path.resolve())
        if env_key not in _LMDB_ENV_CACHE:
            _LMDB_ENV_CACHE[env_key] = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=self.lmdb_path.is_dir(),
            )
        self._env = _LMDB_ENV_CACHE[env_key]
        self._txn = self._env.begin(write=False)

    def _load_images(self, sample_id: str) -> torch.Tensor:
        self._ensure_lmdb_open()
        payload = self._txn.get(sample_id.encode("utf-8"))
        if payload is None:
            raise KeyError(f"Sample not found in LMDB: {sample_id}")

        record = pickle.loads(payload)
        image_bytes_list = record.get("image_bytes", [])
        images = []
        for image_bytes in image_bytes_list[: self.n_views]:
            if image_bytes:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = self.image_transform(img)
            else:
                img = torch.ones(3, self.image_transform.transforms[0].size[0], self.image_transform.transforms[0].size[1])
            images.append(img)

        while len(images) < self.n_views:
            images.append(torch.ones_like(images[0]))
        return torch.stack(images)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        return {
            "sample_id": sample_id,
            "images": self._load_images(sample_id),
            "text_emb": self.text_by_id[sample_id],
        }


def collate_image_latent(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch]),
        "z": torch.stack([item["z"] for item in batch]),
    }


def collate_image_latent_cad(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch]),
        "z": torch.stack([item["z"] for item in batch]),
        "cad_vec": torch.stack([item["cad_vec"] for item in batch]),
    }


def collate_image_only(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch]),
    }


def collate_image_text_latent(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch]),
        "text_emb": torch.stack([item["text_emb"] for item in batch]),
        "z": torch.stack([item["z"] for item in batch]),
    }


def collate_image_text_only(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "images": torch.stack([item["images"] for item in batch]),
        "text_emb": torch.stack([item["text_emb"] for item in batch]),
    }


class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return math.nan
        return self.total / self.count
