import sys
import os
import glob
import gc
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------
repo_path = r"C:\00_SKKUAI\01_2025\01_MedicalAI\02_Exaonepathv1\EXAONEPath"
sys.path.append(repo_path)

# 데이터 루트
dataset_root_folder = r"C:\00_SKKUAI\01_2025\01_MedicalAI\02_Exaonepathv1\EXAONEPath\images\dataset"

# [중요] 저장 폴더 설정 (파일이 여러 개 생깁니다)
output_dir = os.path.join(repo_path, "sae_data_chunks")
os.makedirs(output_dir, exist_ok=True)

# [설정] 약 2GB 단위로 끊어서 저장 (안전함)
TOKENS_PER_CHUNK = 500000 
PATCH_SIZE = 224

print(f"1. 환경 설정")
print(f"   - 저장 폴더: {output_dir}")
print(f"   - 모드: 분할 저장 (RAM 폭발 방지)")

try:
    from macenko import TorchMacenkoNormalizer
    from vision_transformer import VisionTransformer
except ImportError as e:
    sys.exit(f"모듈 에러: {e}")

# ---------------------------------------------------------
# 2. 모델 로드 (RTX 4090)
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"2. 모델 로드 중... ({device})")

model = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath")
model.to(device)
model.eval()
model.requires_grad_(False)

normalizer = TorchMacenkoNormalizer()
transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),              
    transforms.Lambda(lambda x: x * 255) 
])

# ---------------------------------------------------------
# 3. 파일 검색
# ---------------------------------------------------------
image_files = glob.glob(os.path.join(dataset_root_folder, "**", "*.jpg"), recursive=True) + \
              glob.glob(os.path.join(dataset_root_folder, "**", "*.png"), recursive=True)
print(f"   -> 총 {len(image_files)}장 발견")

# ---------------------------------------------------------
# 4. 추출 및 분할 저장
# ---------------------------------------------------------
buffer_vectors = [] 
chunk_idx = 0
total_saved = 0

print("4. 추출 시작")

for img_path in tqdm(image_files, desc="Processing"):
    try:
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        patch_tensors = []
        
        for y in range(0, H, PATCH_SIZE):
            for x in range(0, W, PATCH_SIZE):
                box = (x, y, min(x + PATCH_SIZE, W), min(y + PATCH_SIZE, H))
                patch = img.crop(box)
                if patch.size != (PATCH_SIZE, PATCH_SIZE):
                    new_patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), (255, 255, 255))
                    new_patch.paste(patch, (0, 0))
                    patch = new_patch
                patch_tensors.append(transform_to_tensor(patch))

        if not patch_tensors: continue

        # Macenko
        norm_patches = []
        for pt in patch_tensors:
            try:
                normalized, _, _ = normalizer.normalize(pt, stains=False, form='chw', dtype='float')
                norm_patches.append(normalized)
            except:
                norm_patches.append(pt.float() / 255.0)
        
        batch_input = torch.stack(norm_patches).to(device)

        # Inference (Mixed Precision)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
            outputs = model.get_intermediate_layers(batch_input, n=1)[0]
            flat_features = outputs.reshape(-1, 768)
            buffer_vectors.append(flat_features.float().cpu()) # CPU로 이동

        # [핵심] 버퍼가 차면 저장하고 비우기
        current_len = sum(t.shape[0] for t in buffer_vectors)
        if current_len >= TOKENS_PER_CHUNK:
            chunk_data = torch.cat(buffer_vectors, dim=0)
            chunk_data = chunk_data / (chunk_data.norm(dim=1, keepdim=True) + 1e-6) # 정규화
            
            save_path = os.path.join(output_dir, f"sae_chunk_{chunk_idx:04d}.pt")
            torch.save(chunk_data, save_path)
            
            print(f" [Save] {save_path} ({chunk_data.shape})")
            total_saved += chunk_data.shape[0]
            
            # 메모리 초기화
            buffer_vectors = []
            chunk_idx += 1
            gc.collect()

    except Exception as e:
        continue

# 남은 데이터 저장
if buffer_vectors:
    chunk_data = torch.cat(buffer_vectors, dim=0)
    chunk_data = chunk_data / (chunk_data.norm(dim=1, keepdim=True) + 1e-6)
    save_path = os.path.join(output_dir, f"sae_chunk_{chunk_idx:04d}.pt")
    torch.save(chunk_data, save_path)
    print(f" [Final Save] {save_path}")

print("완료! 이제 train_sae.py를 실행하세요.")