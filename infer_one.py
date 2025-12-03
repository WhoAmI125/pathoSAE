import sys
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------
# 1. 경로 및 설정 (사용자 환경 맞춤)
# ---------------------------------------------------------
# 깃헙 클론된 로컬 폴더 주소
repo_path = r"C:\00_SKKUAI\01_2025\01_MedicalAI\02_Exaonepathv1\EXAONEPath"

# 파이썬이 모듈을 찾을 수 있도록 경로 추가
sys.path.append(repo_path)

# 분석할 큰 이미지 경로 (1000x1000 등)
image_rel_path = "images/1001.jpg" # 여기에 분석할 파일명을 넣으세요
image_full_path = os.path.join(repo_path, image_rel_path)

# 패치 설정
PATCH_SIZE = 224  # EXAONEPath 입력 규격
STRIDE = 224      # 패치 간격 (224면 겹치지 않음, 작으면 겹침)

print(f"1. 환경 설정 완료")
print(f"   - 리포지토리: {repo_path}")
print(f"   - 대상 이미지: {image_full_path}")

# 로컬 모듈 임포트
try:
    from macenko import TorchMacenkoNormalizer
    from vision_transformer import VisionTransformer
    print("   - 모듈 임포트 성공")
except ImportError as e:
    print(f"\n[오류] 필수 모듈을 찾을 수 없습니다.\n에러: {e}")
    sys.exit()

# ---------------------------------------------------------
# 2. 이미지 로드 및 패치 분할 (Sliding Window)
# ---------------------------------------------------------
if not os.path.exists(image_full_path):
    print(f"\n[오류] 파일이 없습니다: {image_full_path}")
    sys.exit()

# 이미지 열기
original_large_img = Image.open(image_full_path).convert('RGB')
W, H = original_large_img.size
print(f"2. 원본 이미지 로드 완료 ({W}x{H}) -> 패치 분할 시작...")

# 텐서 변환기 (0~255 범위 유지)
transform_to_tensor = transforms.Compose([
    transforms.ToTensor(),              
    transforms.Lambda(lambda x: x * 255) 
])

patch_tensors = []  # 모델에 들어갈 텐서들
patch_images = []   # 시각화용 원본 패치 이미지들
patch_coords = []   # 패치 위치 (x, y)

# 이중 루프로 이미지 자르기
for y in range(0, H, STRIDE):
    for x in range(0, W, STRIDE):
        # 영역 설정
        box = (x, y, min(x + PATCH_SIZE, W), min(y + PATCH_SIZE, H))
        patch = original_large_img.crop(box)
        
        # 패딩 (Padding): 자투리 이미지가 224보다 작으면 검은색으로 채움
        if patch.size != (PATCH_SIZE, PATCH_SIZE):
            new_patch = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), (0, 0, 0))
            new_patch.paste(patch, (0, 0)) # 왼쪽 상단에 붙여넣기
            patch = new_patch
        
        patch_images.append(patch)
        patch_tensors.append(transform_to_tensor(patch))
        patch_coords.append((x, y))

print(f"   - 패치 분할 완료: 총 {len(patch_tensors)}개의 패치 생성")

# ---------------------------------------------------------
# 3. Macenko Normalization (배치 처리)
# ---------------------------------------------------------
normalizer = TorchMacenkoNormalizer()
norm_patch_list = []

print("3. Macenko Normalization 진행 중 (시간이 조금 걸릴 수 있습니다)...")
for i, pt in enumerate(patch_tensors):
    try:
        # 각 패치 정규화
        normalized, _, _ = normalizer.normalize(pt, stains=False, form='chw', dtype='float')
        norm_patch_list.append(normalized)
    except Exception:
        # 실패 시 단순 스케일링
        norm_patch_list.append(pt.float() / 255.0)

# 리스트를 하나의 배치 텐서로 결합: (N, 3, 224, 224)
batch_input = torch.stack(norm_patch_list)
print(f"   - 정규화 완료. 최종 입력 데이터 크기: {batch_input.shape}")

# ---------------------------------------------------------
# 4. EXAONEPath 모델 로드
# ---------------------------------------------------------
print("4. 모델 로드 중... (Hugging Face)")
try:
    model = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath")
    model.eval()
    print("   - 모델 로드 성공")
except Exception as e:
    print(f"[오류] 모델 로드 실패: {e}")
    sys.exit()

# ---------------------------------------------------------
# 5. 추론 및 특징 추출 (Inference)
# ---------------------------------------------------------
print("5. 특징 추출 및 분석 시작...")
with torch.no_grad():
    # 전체 패치를 한 번에 모델에 입력 (Batch Processing)
    # A. 특징 벡터 (N, 768)
    features = model(batch_input)
    
    # B. 어텐션 맵 (N, 12, 197, 197)
    attentions = model.get_last_selfattention(batch_input)

print(f"   - 분석 완료!")
print(f"   - [결과 1] 전체 특징 벡터 행렬: {features.shape}") 
# 예: (25, 768) -> 25개의 패치가 각각 768개의 특징을 가짐

# ---------------------------------------------------------
# 6. 결과 시각화 (예시: 중앙에 있는 패치 하나 확인)
# ---------------------------------------------------------
# 시각화할 패치 인덱스 선택 (중간쯤 있는 패치)
sample_idx = len(patch_tensors) // 2 
x, y = patch_coords[sample_idx]

print(f"6. 시각화: {sample_idx}번째 패치 (위치: x={x}, y={y})의 결과를 보여줍니다.")

# 해당 패치의 어텐션 맵 가져오기
att_map = torch.mean(attentions[sample_idx], dim=0) # 헤드 평균
cls_att = att_map[0, 1:] # CLS 토큰과 나머지 간의 관계
grid_size = int(np.sqrt(cls_att.shape[0]))
att_numpy = cls_att.reshape(grid_size, grid_size).numpy()

# 어텐션 맵 리사이즈 (224x224)
att_resized = cv2.resize(att_numpy, (PATCH_SIZE, PATCH_SIZE))

# 정규화된 이미지 (시각화용 변환)
norm_img_show = batch_input[sample_idx].permute(1, 2, 0).numpy()
norm_img_show = np.clip(norm_img_show, 0, 1)

# 그리기
plt.figure(figsize=(12, 4))

# 원본 패치
plt.subplot(1, 3, 1)
plt.imshow(patch_images[sample_idx])
plt.title(f"Original Patch\n(x={x}, y={y})")
plt.axis('off')

# 정규화된 패치
plt.subplot(1, 3, 2)
plt.imshow(norm_img_show)
plt.title("Macenko Normalized")
plt.axis('off')

# 어텐션 맵
plt.subplot(1, 3, 3)
plt.imshow(patch_images[sample_idx])
plt.imshow(att_resized, cmap='jet', alpha=0.5)
plt.title("Self-Attention Map")
plt.axis('off')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. (추가) 전체 이미지 어텐션 맵 이어붙이기 (Stitching)
# ---------------------------------------------------------
print("7. 전체 이미지 어텐션 맵 복원 중...")

# 1) 빈 캔버스 만들기 (원본 이미지 크기: H, W)
full_attention_map = np.zeros((H, W), dtype=np.float32)
# 겹치는 부분의 평균을 구하기 위해 카운트 맵도 생성
count_map = np.zeros((H, W), dtype=np.float32)

# 2) 모든 패치를 순회하며 제자리에 붙여넣기
for idx, (x, y) in enumerate(patch_coords):
    # 해당 패치의 어텐션 맵 가져오기
    att_tensor = torch.mean(attentions[idx], dim=0) # 헤드 평균
    cls_att = att_tensor[0, 1:] # CLS 토큰 제외
    
    # 14x14 크기를 224x224로 확대
    grid_size = int(np.sqrt(cls_att.shape[0]))
    att_numpy = cls_att.reshape(grid_size, grid_size).numpy()
    att_resized = cv2.resize(att_numpy, (PATCH_SIZE, PATCH_SIZE))
    
    # 원본 이미지 캔버스에 붙여넣기
    # (이미지 끝부분 처리를 위해 크기 계산)
    h_paste = min(PATCH_SIZE, H - y)
    w_paste = min(PATCH_SIZE, W - x)
    
    full_attention_map[y:y+h_paste, x:x+w_paste] += att_resized[:h_paste, :w_paste]
    count_map[y:y+h_paste, x:x+w_paste] += 1

# 3) 평균 내기 (겹치는 부분이 있다면)
full_attention_map /= np.maximum(count_map, 1)

# 4) 시각화 (원본 위에 오버레이)
# 어텐션 맵 정규화 (0~1)
full_attention_map = (full_attention_map - full_attention_map.min()) / (full_attention_map.max() - full_attention_map.min())
full_attention_map = np.uint8(255 * full_attention_map)

# 히트맵 컬러 적용
heatmap = cv2.applyColorMap(full_attention_map, cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# 원본 이미지를 numpy로 변환
original_np = np.array(original_large_img)

# 합성 (원본 60% + 히트맵 40%)
overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)

# 출력
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.title(f"Whole Slide Attention Map ({W}x{H})")
plt.axis('off')
plt.show()

# ---------------------------------------------------------
# 7. (추가) SAE 테스트를 위한 중간 Layer 추출 및 저장
# ---------------------------------------------------------
print("-" * 50)
print("7. SAE(Sparse Autoencoder) 입력 데이터 생성 테스트")

# 1) 중간 Layer 추출 (Layer 12)
# n=1 : 마지막 1개 블록의 출력을 가져옴
# 반환값은 리스트 형태이므로 [0]으로 꺼내야 함
intermediate_output = model.get_intermediate_layers(batch_input, n=1)[0]

print(f"   - [1단계] 추출된 Raw Data 크기: {intermediate_output.shape}")
# 예상: [25, 197, 768] 
# (25: 패치 개수, 197: 토큰 개수(1 CLS + 196 Patch), 768: 차원)

# 2) 데이터 펼치기 (Flattening)
# SAE는 보통 [Batch, Dimension] 형태의 2D 행렬을 입력으로 받습니다.
# 따라서 (패치개수 x 토큰개수)를 하나의 거대한 배치로 합칩니다.
sae_input_data = intermediate_output.reshape(-1, 768)

print(f"   - [2단계] SAE 입력용 변환 크기: {sae_input_data.shape}")
# 예상: [4925, 768] (25 * 197 = 4925)

# 3) 데이터 저장 (SAE 학습 코드에서 불러오기 위해)
save_path = os.path.join(repo_path, "sae_test_data.pt")
torch.save(sae_input_data, save_path)

print(f"   - [3단계] 저장 완료: {save_path}")
print(f"   -> 이제 이 파일을 로드해서 SAE가 에러 없이 돌아가는지 확인하세요.")