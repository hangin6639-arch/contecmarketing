"""
Contec 3-Layer Semantic Distortion Flow Analyzer (Phase 7-C)
======================================================================
목표: Identity(L1) -> PR(L2) -> Market(L3) 3중 구조 레이더 차트 시각화 및 
의미 왜곡 발생 구간(CASE A vs B) 자동 진단 로직 탑재.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("../outputs/strategy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[1/5] SBERT 모델 로드 중...")
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# ── 1. 데이터 로드 및 Layer 분류 ──────────────────────────────
print("[2/5] 3-Layer 코퍼스 데이터 로드 중...")
try:
    corpus_df = pd.read_csv("../outputs/identity/contec_brand_corpus.csv")
    layer1_texts = corpus_df[corpus_df['layer'] == 'Layer_1_Identity']['sentence'].tolist()
    layer2_texts = corpus_df[corpus_df['layer'] == 'Layer_2_Curated_Image']['sentence'].tolist()
    
    raw_df = pd.read_csv("../outputs/phase5_normalized_space.csv")
    layer3_texts = raw_df['purified_body'].tolist()
except FileNotFoundError as e:
    print(f"🚨 [오류] 데이터 파일을 찾을 수 없습니다: {e}")
    exit()

# ── 2. Layer별 임베딩 및 Centroid 계산 ───────────
print("[3/5] Layer별 의미 공간 벡터 계산 중...")
centroid_L1 = np.mean(model.encode(layer1_texts), axis=0).reshape(1, -1) if layer1_texts else np.zeros((1, 768))
centroid_L2 = np.mean(model.encode(layer2_texts), axis=0).reshape(1, -1) if layer2_texts else np.zeros((1, 768))
centroid_L3 = np.mean(model.encode(layer3_texts), axis=0).reshape(1, -1) if layer3_texts else np.zeros((1, 768))

# ── 3. 4대 핵심 Anchor 투영 (Projection) ───────────────────
brand_anchors = {
    "GSaaS & 인프라\n(지상국 네트워크)": "전 세계 글로벌 지상국 네트워크 구축, GSaaS 우주 인프라, 광학 지상국 OGS, 우주 상황 인식 SSA",
    "Space Data Pipeline\n(데이터 수신/분석)": "위성 영상 원시 데이터 수신, 데이터 전처리 보정, 딥러닝 기반 위성 데이터 분석 및 올인원 활용 서비스",
    "Hardware Ecosystem\n(AP위성/수직계열화)": "AP위성 인수, 위성 통신 단말기, 인공위성 탑재체 제조, 시스템 수직 계열화",
    "Policy & Theme\n(정책 수혜/테마)": "우주항공청 수혜주, 대전 우주산업 클러스터, 주가 급등, 상장, 목표가, 거래량"
}
anchor_embs = [model.encode([v])[0] for v in brand_anchors.values()]
labels = list(brand_anchors.keys())

# L1, L2, L3 투영도 계산
proj_L1 = cosine_similarity(centroid_L1, anchor_embs).flatten()
proj_L2 = cosine_similarity(centroid_L2, anchor_embs).flatten()
proj_L3 = cosine_similarity(centroid_L3, anchor_embs).flatten()

# ── 4. 자동 해석 엔진 (Distortion Logic) ───────────────────
print("[4/5] 의미 왜곡 구간 자동 진단 중...\n")

pr_align_score = cosine_similarity(centroid_L1, centroid_L2)[0][0]
market_align_score = cosine_similarity(centroid_L2, centroid_L3)[0][0]

print("="*60)
print("📊 [Layer 2 (Curated PR) 의도 분석]")
for idx, label in enumerate(labels):
    print(f" - {label.split()[-1]}: {proj_L2[idx]:.3f}")
top_pr_anchor = labels[np.argmax(proj_L2)].replace('\n', ' ')
print(f" ➡️ 공식 PR에서 가장 강조된 메시지: [{top_pr_anchor}]\n")

print("🔍 [의미 왜곡 흐름 자동 진단 결과]")
# 임계값(Threshold) 설정 (예: 0.65 이상이면 구조가 유사하다고 판단)
THRESHOLD = 0.65

if pr_align_score >= THRESHOLD:
    if market_align_score < THRESHOLD:
        print("🚨 [CASE A 도출] Identity ≈ PR 이지만 PR ≠ Market")
        print(" ➡️ 해석: 기업은 본질(데이터/인프라)에 맞게 PR을 전개했으나, 시장/언론이 이를 받아들이는 과정에서 '정책·테마' 등으로 프레이밍을 심각하게 왜곡(Distortion)시키고 있습니다. PR 채널 다각화 및 통제력 강화가 필요합니다.")
    else:
        print("✅ [CASE C 도출] Identity ≈ PR ≈ Market")
        print(" ➡️ 해석: 훌륭한 브랜드 커뮤니케이션. 기업의 본질이 PR을 거쳐 시장까지 성공적으로 전달되고 있습니다.")
else:
    print("🚨 [CASE B 도출] Identity ≠ PR 부터 불일치 발생")
    print(f" ➡️ 해석: 기업 본질(L1)과 PR 메시지(L2)가 분리되어 있습니다. 언론의 탓을 하기 전, 기업의 PR 전략 자체가 본질(기술)보다 이슈({top_pr_anchor}) 중심으로 편향 설계되었을 가능성이 높습니다.")
print("="*60)

# ── 5. 3-Layer 레이더 차트 시각화 ──────────────────────────
print("\n[5/5] 3-Layer 흐름 시각화 생성 중...")
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

scores_L1 = proj_L1.tolist() + [proj_L1[0]]
scores_L2 = proj_L2.tolist() + [proj_L2[0]]
scores_L3 = proj_L3.tolist() + [proj_L3[0]]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# [Layer 3] Raw Market (시장 담론) - 빨강 (Solid)
ax.plot(angles, scores_L3, color='#e74c3c', linewidth=2.5, linestyle='solid', label='Layer 3: Market (시장 소비)')
ax.fill(angles, scores_L3, color='#e74c3c', alpha=0.15)

# [Layer 2] Curated PR (공식 PR) - 파랑 (Dash-dot)
ax.plot(angles, scores_L2, color='#3498db', linewidth=2.5, linestyle='dashdot', label='Layer 2: PR/Newsroom (기업 의도)')
# L2는 흐름을 보기 위해 채색(Fill)은 생략하거나 투명도 극소화

# [Layer 1] Internal Identity (기업 본질) - 초록 (Dashed)
ax.plot(angles, scores_L1, color='#2ecc71', linewidth=2.5, linestyle='dashed', label='Layer 1: Identity (기업 본질)')
ax.fill(angles, scores_L1, color='#2ecc71', alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color='#333')
ax.set_ylim(min(min(scores_L1), min(scores_L2), min(scores_L3)) - 0.05, max(max(scores_L1), max(scores_L2), max(scores_L3)) + 0.05)

plt.title("Semantic Distortion Flow (Identity ➡️ PR ➡️ Market)", y=1.12, fontsize=16, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
plt.tight_layout()

output_file = OUTPUT_DIR / "brand_communication_flow.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"🎉 [실행 완료] outputs/strategy 폴더에 [{output_file.name}] 가 저장되었습니다.")