"""
Contec 3-Layer Brand Gap Analyzer (Phase 7-B)
======================================================================
목표: Identity(Layer 1), Curated(Layer 2), Raw Market(Layer 3) 간의 
의미적 왜곡을 계산하고, 4대 핵심 가치에 대한 투영 갭(Gap)을 시각화합니다.
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

print("[1/4] SBERT 모델 로드 중...")
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# ── 1. 데이터 로드 및 Layer 분류 ──────────────────────────────
print("[2/4] 3-Layer 코퍼스 데이터 로드 중...")
# Layer 1 & 2 (방금 추출한 공식 코퍼스)
corpus_df = pd.read_csv("../outputs/identity/contec_brand_corpus.csv")
layer1_texts = corpus_df[corpus_df['layer'] == 'Layer_1_Identity']['sentence'].tolist()
layer2_texts = corpus_df[corpus_df['layer'] == 'Layer_2_Curated_Image']['sentence'].tolist()

# Layer 3 (기존에 정제해둔 네이버 뉴스 실제 담론)
raw_df = pd.read_csv("../outputs/phase5_normalized_space.csv")
layer3_texts = raw_df['purified_body'].tolist()

# ── 2. Layer별 임베딩 및 Centroid(대표 벡터) 계산 ───────────
print("[3/4] Layer별 의미 공간(Semantic Space) 및 왜곡도 계산 중...")
emb_L1 = model.encode(layer1_texts) if layer1_texts else np.zeros((1, 768))
emb_L2 = model.encode(layer2_texts) if layer2_texts else np.zeros((1, 768))
emb_L3 = model.encode(layer3_texts) if layer3_texts else np.zeros((1, 768))

# 각 Layer의 중심 좌표(평균 벡터) 도출
centroid_L1 = np.mean(emb_L1, axis=0).reshape(1, -1)
centroid_L2 = np.mean(emb_L2, axis=0).reshape(1, -1)
centroid_L3 = np.mean(emb_L3, axis=0).reshape(1, -1)

# 왜곡도 계산 (1 - Cosine Similarity)
pr_alignment = cosine_similarity(centroid_L1, centroid_L2)[0][0]
market_distortion = 1 - cosine_similarity(centroid_L2, centroid_L3)[0][0]
final_brand_gap = 1 - cosine_similarity(centroid_L1, centroid_L3)[0][0]

print("\n" + "="*50)
print("📊 [3-Layer 브랜드 의미 왜곡도 분석 결과]")
print(f"✔️ PR 정렬도 (Identity ↔ Curated Image) : {pr_alignment:.3f} (1에 가까울수록 공식 메시지가 PR에 잘 반영됨)")
print(f"🚨 시장 왜곡도 (Curated Image ↔ Raw Market): {market_distortion:.3f} (0에 가까울수록 왜곡 없음, 높을수록 굴절 심함)")
print(f"🔥 최종 브랜드 갭 (Identity ↔ Raw Market) : {final_brand_gap:.3f} (기업의 본질과 시장 인식 간의 최종 격차)")
print("="*50 + "\n")

# ── 3. 4대 핵심 Anchor 기반 Identity vs Image 갭 시각화 ─────
print("[4/4] 마케팅 전략 도출용 Gap Analysis 레이더 차트 생성 중...")

brand_anchors = {
    "GSaaS & 인프라\n(지상국 네트워크)": "전 세계 글로벌 지상국 네트워크 구축, GSaaS 우주 인프라, 광학 지상국 OGS, 우주 상황 인식 SSA",
    "Space Data Pipeline\n(데이터 수신/분석)": "위성 영상 원시 데이터 수신, 데이터 전처리 보정, 딥러닝 기반 위성 데이터 분석 및 올인원 활용 서비스",
    "Hardware Ecosystem\n(AP위성/수직계열화)": "AP위성 인수, 위성 통신 단말기, 인공위성 탑재체 제조, 시스템 수직 계열화",
    "Policy & Theme\n(정책 수혜 및 테마)": "우주항공청 수혜주, 대전 우주산업 클러스터, 주가 급등, 상장, 목표가, 거래량"
}

anchor_embs = [model.encode([v])[0] for v in brand_anchors.values()]

# Layer 1(본질)과 Layer 3(시장)가 각각 4대 Anchor에 얼마나 투영되는지 계산
l1_projection = cosine_similarity(centroid_L1, anchor_embs).flatten()
l3_projection = cosine_similarity(centroid_L3, anchor_embs).flatten()

# 시각화
labels = list(brand_anchors.keys())
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

l1_scores = l1_projection.tolist() + [l1_projection[0]]
l3_scores = l3_projection.tolist() + [l3_projection[0]]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 시장 인식 (Raw Image)
ax.plot(angles, l3_scores, color='#e74c3c', linewidth=2.5, linestyle='solid', label='Raw Market (시장 인식)')
ax.fill(angles, l3_scores, color='#e74c3c', alpha=0.2)

# 기업 본질 (Internal Identity)
ax.plot(angles, l1_scores, color='#2ecc71', linewidth=2.5, linestyle='dashed', label='Internal Identity (기업 본질)')
ax.fill(angles, l1_scores, color='#2ecc71', alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color='#333')
ax.set_ylim(min(min(l1_scores), min(l3_scores)) - 0.05, max(max(l1_scores), max(l3_scores)) + 0.05)

plt.title("Contec Brand Gap Analysis (Identity vs Market)", y=1.1, fontsize=16, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

output_file = OUTPUT_DIR / "brand_gap_radar.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"🎉 [분석 완료] outputs/strategy 폴더에 [{output_file.name}] 시각화가 저장되었습니다.")
print("💡 출력된 왜곡도 수치와 레이더 차트의 Gap 면적을 확인해 주세요!")