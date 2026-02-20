from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchstain.base.normalizers.he_normalizer import HENormalizer
    from torchstain.torch.utils import cov, percentile
except Exception as exc:  # pragma: no cover - dependency is optional at import time
    raise ImportError(
        "torchstain is required for Macenko normalization. Install `torchstain>=1.3.0`."
    ) from exc


class TorchMacenkoNormalizer(HENormalizer):
    """Torch implementation of Macenko normalization."""

    def __init__(self):
        super().__init__()
        self.HERef = torch.tensor(
            [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]],
            dtype=torch.float32,
        )
        self.maxCRef = torch.tensor([1.9705, 1.0308], dtype=torch.float32)
        self.updated_lstsq = hasattr(torch.linalg, "lstsq")

    def __convert_rgb2od(self, image: torch.Tensor, Io: int, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.permute(1, 2, 0)

        od = -torch.log((image.reshape((-1, image.shape[-1])).float() + 1) / Io)
        odhat = od[~torch.any(od < beta, dim=1)]
        return od, odhat

    def __find_HE(self, odhat: torch.Tensor, eigvecs: torch.Tensor, alpha: float) -> torch.Tensor:
        that = torch.matmul(odhat, eigvecs)
        phi = torch.atan2(that[:, 1], that[:, 0])

        min_phi = percentile(phi, alpha)
        max_phi = percentile(phi, 100 - alpha)

        v_min = torch.matmul(
            eigvecs,
            torch.stack((torch.cos(min_phi), torch.sin(min_phi))),
        ).unsqueeze(1)
        v_max = torch.matmul(
            eigvecs,
            torch.stack((torch.cos(max_phi), torch.sin(max_phi))),
        ).unsqueeze(1)

        return torch.where(
            v_min[0] > v_max[0],
            torch.cat((v_min, v_max), dim=1),
            torch.cat((v_max, v_min), dim=1),
        )

    def __find_concentration(self, od: torch.Tensor, HE: torch.Tensor) -> torch.Tensor:
        y = od.T
        if not self.updated_lstsq:
            return torch.lstsq(y, HE)[0][:2]
        return torch.linalg.lstsq(HE, y)[0]

    def __compute_matrices(
        self,
        image: torch.Tensor,
        Io: int,
        alpha: float,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        od, odhat = self.__convert_rgb2od(image, Io=Io, beta=beta)

        _, eigvecs = torch.linalg.eigh(cov(odhat.T))
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(odhat, eigvecs, alpha)
        C = self.__find_concentration(od, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
        return HE, C, maxC

    def fit(self, image: torch.Tensor, Io: int = 240, alpha: float = 1, beta: float = 0.15):
        HE, _, maxC = self.__compute_matrices(image, Io, alpha, beta)
        self.HERef = HE
        self.maxCRef = maxC

    def normalize(
        self,
        image: torch.Tensor,
        Io: int = 240,
        alpha: float = 1,
        beta: float = 0.15,
        stains: bool = True,
        form: str = "chw",
        dtype: str = "int",
    ):
        c, h, w = image.shape

        HE, C, maxC = self.__compute_matrices(image, Io, alpha, beta)
        C *= (self.maxCRef / maxC).unsqueeze(-1)

        inorm = Io * torch.exp(-torch.matmul(self.HERef, C))
        inorm[inorm > 255] = 255

        if form == "chw" and dtype == "int":
            inorm = inorm.reshape(c, h, w).int()
        elif form == "chw" and dtype == "float":
            inorm = inorm.reshape(c, h, w).float() / 255.0
        elif form == "hwc" and dtype == "int":
            inorm = inorm.T.reshape(h, w, c).int()
        elif form == "hwc" and dtype == "float":
            inorm = inorm.T.reshape(h, w, c).float() / 255.0
        else:
            raise ValueError("check macenko input form and dtype")

        H, E = None, None
        if stains:
            H = torch.mul(Io, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()

            E = torch.mul(Io, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()

        return inorm, H, E


class MacenkoNormalizer:
    """Callable Macenko normalizer used in extraction pipeline."""

    def __init__(self, target_path: str | None = None):
        self.transform_before_macenko = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255),
            ]
        )
        self.normalizer = TorchMacenkoNormalizer()

        if target_path:
            path = Path(target_path)
            if not path.exists():
                raise FileNotFoundError(f"Macenko target image not found: {path}")

            target = Image.open(path).convert("RGB")
            self.normalizer.fit(self.transform_before_macenko(target))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_tensor = self.transform_before_macenko(image)

        try:
            normalized, _, _ = self.normalizer.normalize(
                image=image_tensor,
                stains=False,
                form="chw",
                dtype="float",
            )
            if torch.any(torch.isnan(normalized)):
                return transforms.ToTensor()(image)
            return normalized
        except Exception:
            return transforms.ToTensor()(image)


# Backward-compatible alias used in pathoSAEv2 scripts.
macenko_normalizer = MacenkoNormalizer
