import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathoSAE.sae_model import SparseAutoencoder, SAEConfig
from macenko import TorchMacenkoNormalizer
from vision_transformer import VisionTransformer

# ---------------------------------------------------------
# 1. 설정 (경로 및 이미지 지정)
# ---------------------------------------------------------
repo_path = r"C:\00_SKKUAI\01_2025\01_MedicalAI\02_Exaonepathv1\EXAONEPath"

# [수정 필요] 분석할 이미지 경로를 입력하세요. (TCGA 데이터 중 하나)
img_path = os.path.join(repo_path, "images/dataset/TCGA-A1-A0SE/308.jpg") 
# 예시: img_path = r"C:\...\EXAONEPath\images\dataset\TCGA-A2-A0T2\...\sample_image.png"

# 학습된 모델 경로
sae_path = os.path.join(repo_path, "sae_model_checkpoint.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 224

# ---------------------------------------------------------
# 2. 모델 로드
# ---------------------------------------------------------
print("모델 로드 중...")

# A. EXAONEPath (Backbone)
backbone = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath")
backbone.to(device).eval()

# B. SAE (Feature Extractor)
cfg = SAEConfig(d_in=768, expansion_factor=32)
sae = SparseAutoencoder(cfg)

if not os.path.exists(sae_path):
    print("오류: sae_model_checkpoint.pth 파일이 없습니다.")
    exit()

sae.load_state_dict(torch.load(sae_path, map_location=device))
sae.to(device).eval()

print("   -> 로드 완료!")

# ---------------------------------------------------------
# 3. 이미지 처리 및 추론
# ---------------------------------------------------------
if not os.path.exists(img_path):
    print(f"오류: 이미지 파일을 찾을 수 없습니다.\n경로: {img_path}")
    exit()

# 이미지 열기
original_img = Image.open(img_path).convert('RGB')
W, H = original_img.size
print(f"이미지 분석 중... ({W}x{H})")

# 전처리 도구
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])
normalizer = TorchMacenkoNormalizer()

# 패치 단위로 잘라서 리스트에 담기
patch_coords = []
patch_tensors = []

for y in range(0, H, PATCH_SIZE):
    for x in range(0, W, PATCH_SIZE):
        box = (x, y, min(x + PATCH_SIZE, W), min(y + PATCH_SIZE, H))
        patch = original_img.crop(box)
        
        # 패딩 (흰색)
        if patch.size != (PATCH_SIZE, PATCH_SIZE):
            new_patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), (255, 255, 255))
            new_patch.paste(patch, (0, 0))
            patch = new_patch
        
        pt = transform_tensor(patch)
        
        # Macenko Normalization
        try:
            norm_pt, _, _ = normalizer.normalize(pt, stains=False, form='chw', dtype='float')
        except:
            norm_pt = pt.float() / 255.0
            
        patch_tensors.append(norm_pt)
        patch_coords.append((x, y))

# 배치 추론
batch_input = torch.stack(patch_tensors).to(device)

with torch.no_grad():
    # 1. ViT 특징 추출
    vit_out = backbone.get_intermediate_layers(batch_input, n=1)[0]
    # [N_patches, 197, 768] -> [N_patches, 768] (전체 토큰 평균 사용)
    features = vit_out.mean(dim=1)
    # 정규화
    features = features / (features.norm(dim=1, keepdim=True) + 1e-6)
    
    # 2. SAE 통과
    sae_out = sae(features)
    activations = sae_out["feature_acts"] # [N_patches, 24576]

# ---------------------------------------------------------
# 4. 가장 강력한 특징 찾기 & 시각화
# ---------------------------------------------------------
# 이 이미지 전체에서 가장 강하게 활성화된 뉴런 TOP 1 찾기
# (Max over patches)
max_acts_per_neuron = activations.max(dim=0)[0] # [24576]
top_neuron_idx = torch.argmax(max_acts_per_neuron).item()
top_neuron_val = max_acts_per_neuron[top_neuron_idx].item()

print(f"\n[분석 결과]")
print(f"이 이미지에서 가장 두드러진 특징(Concept)은 **Neuron #{top_neuron_idx}** 입니다.")
print(f"최대 활성도: {top_neuron_val:.4f}")

# 히트맵 그리기
heatmap = np.zeros((H, W), dtype=np.float32)

# 해당 뉴런의 패치별 활성도 가져오기
neuron_activations = activations[:, top_neuron_idx].cpu().numpy()

for idx, (x, y) in enumerate(patch_coords):
    score = neuron_activations[idx]
    # 패치 영역에 점수 채우기
    h_paste = min(PATCH_SIZE, H - y)
    w_paste = min(PATCH_SIZE, W - x)
    heatmap[y:y+h_paste, x:x+w_paste] = score

# 시각화
plt.figure(figsize=(12, 6))

# 원본
plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis('off')

# 히트맵 오버레이
plt.subplot(1, 2, 2)
plt.imshow(original_img, alpha=0.6)
# 정규화 후 컬러맵 적용
heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
plt.imshow(heatmap_norm, cmap='jet', alpha=0.5) 
plt.title(f"Concept #{top_neuron_idx} Activation Map")
plt.axis('off')

plt.tight_layout()
plt.show()