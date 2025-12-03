import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class SAEConfig:
    def __init__(self, d_in=768, expansion_factor=32, l1_coefficient=0.005):
        self.d_in = d_in
        self.d_sae = d_in * expansion_factor # 은닉층 크기 (보통 32배~64배)
        self.l1_coefficient = l1_coefficient
        self.dtype = torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

class SparseAutoencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = cfg.d_sae
        self.l1_coeff = cfg.l1_coefficient
        self.device = cfg.device

        # 1. Encoder (W_enc, b_enc)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(self.d_in, self.d_sae, dtype=cfg.dtype, device=self.device)
        ))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae, dtype=cfg.dtype, device=self.device))

        # 2. Decoder (W_dec, b_dec)
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(self.d_sae, self.d_in, dtype=cfg.dtype, device=self.device)
        ))
        
        # Decoder 가중치 정규화 (Unit Norm) - Anthropic 방식
        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=cfg.dtype, device=self.device))
        
        self.to(self.device)

    def forward(self, x):
        # x: [Batch, d_in]
        
        # A. Pre-Bias 제거 (Anthropic 방식)
        x_cent = x - self.b_dec

        # B. Encoder: Linear -> ReLU
        # (Batch, d_in) @ (d_in, d_sae) -> (Batch, d_sae)
        feature_acts = F.relu(x_cent @ self.W_enc + self.b_enc)

        # C. Decoder: Linear
        # (Batch, d_sae) @ (d_sae, d_in) -> (Batch, d_in)
        reconstructed = feature_acts @ self.W_dec + self.b_dec

        # D. Loss 계산
        # 1) MSE (복원 오차)
        # 차원별 정규화된 MSE 사용 (제공된 코드 로직 유지)
        per_item_mse = (reconstructed - x).pow(2).sum(dim=1) 
        # normalization_scale = (x.pow(2).sum(dim=1).sqrt()) # 원본 코드 방식 (L2 norm)
        # mse_loss = (per_item_mse / normalization_scale).mean()
        mse_loss = per_item_mse.mean() # 간단하게 일반 MSE 사용 (초기 테스트용)

        # 2) L1 (희소성 오차) - 활성화된 뉴런의 합
        l1_loss = self.l1_coeff * feature_acts.sum(dim=1).mean()

        loss = mse_loss + l1_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "reconstructed": reconstructed,
            "feature_acts": feature_acts
        }
    
    @torch.no_grad()
    def make_decoder_weights_normalized(self):
        # 학습 도중 Decoder의 크기가 커지는 것을 방지하기 위해 정규화
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)