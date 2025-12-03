import torch
import torch.optim as optim
from pathoSAE.sae_model import SparseAutoencoder, SAEConfig
import matplotlib.pyplot as plt
import os
import glob
import random

# ---------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------
repo_path = r"C:\00_SKKUAI\01_2025\01_MedicalAI\02_Exaonepathv1\EXAONEPath"
data_folder = os.path.join(repo_path, "sae_data_chunks")

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 데이터 파일 목록 가져오기
chunk_files = sorted(glob.glob(os.path.join(data_folder, "*.pt")))
if not chunk_files:
    print("오류: 데이터 파일이 없습니다. generate_sae_dataset.py를 먼저 실행하세요.")
    exit()

print(f"총 {len(chunk_files)}개의 데이터 청크를 찾았습니다.")

# ---------------------------------------------------------
# 2. 모델 초기화 (안정성 강화)
# ---------------------------------------------------------
# [수정 1] 학습률(LR) 하향 조정 (1e-3 -> 4e-4)
# 초기 학습이 너무 빠르면 발산할 수 있습니다.
learning_rate = 4e-4 
cfg = SAEConfig(d_in=768, expansion_factor=32, l1_coefficient=3e-4) 
sae = SparseAutoencoder(cfg)
sae.to(device)
optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

# ---------------------------------------------------------
# 3. 학습 루프 (Safety Mode ON)
# ---------------------------------------------------------
print(f"학습 시작... (LR: {learning_rate}, Gradient Clipping: ON)")
losses = []
epochs = 5 

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    random.shuffle(chunk_files) # 데이터 순서 섞기
    
    for i, fpath in enumerate(chunk_files):
        # A. 청크 로드
        try:
            data = torch.load(fpath, map_location=device)
        except Exception as e:
            print(f" [Skip] 파일 로드 에러: {os.path.basename(fpath)}")
            continue
            
        # [수정 2] 데이터 세탁 (NaN/Inf 제거)
        # Chunk 26번 같은 곳에 이상한 값이 있어도 여기서 0으로 바뀝니다.
        if torch.isnan(data).any() or torch.isinf(data).any():
            # print(f" [Warning] {os.path.basename(fpath)} 데이터 오염 감지 -> 자동 수정됨")
            data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

        # B. 배치 학습
        batch_size = 4096
        num_batches = data.shape[0] // batch_size
        chunk_loss = 0
        valid_batches = 0
        
        # 데이터 셔플
        perm = torch.randperm(data.shape[0], device=device)
        data = data[perm]
        
        for j in range(0, data.shape[0], batch_size):
            batch = data[j : j + batch_size]
            if batch.shape[0] < 10: continue 
            
            optimizer.zero_grad()
            
            # 순전파
            output = sae(batch)
            loss = output["loss"]
            
            # [수정 3] Loss가 NaN이면 비상 탈출 (모델 오염 방지)
            if torch.isnan(loss):
                print(f" [Alert] Batch Loss NaN 발생! 이 배치는 건너뜁니다.")
                continue

            # 역전파
            loss.backward()
            
            # [수정 4] Gradient Clipping (가장 중요!)
            # 기울기가 1.0을 넘으면 강제로 잘라내어 폭발 방지
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            
            # Decoder 가중치 정규화 & 업데이트
            sae.make_decoder_weights_normalized()
            optimizer.step()
            
            chunk_loss += loss.item()
            valid_batches += 1
        
        # 로그 출력
        if valid_batches > 0:
            avg_loss = chunk_loss / valid_batches
            losses.append(avg_loss)
            print(f" Epoch {epoch+1} | Chunk {i+1}/{len(chunk_files)} | Loss: {avg_loss:.6f}")
        
        # 메모리 해제
        del data
        torch.cuda.empty_cache()

# ---------------------------------------------------------
# 4. 결과 저장
# ---------------------------------------------------------
print("학습 완료!")
save_path = os.path.join(repo_path, "sae_model_checkpoint.pth")
torch.save(sae.state_dict(), save_path)
print(f"모델 저장됨: {save_path}")

if len(losses) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("SAE Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show()