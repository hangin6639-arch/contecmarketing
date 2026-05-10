"""
컨텍(Contec) Semantic Geometry 분석 파이프라인 (Phase 5: Normalization & t-SNE)
========================================================================================
목표: 의미 공간의 왜곡(중복 편향, 점수 포화)을 제거하고, t-SNE를 통해 컨텍의 실제 좌표를 투영.
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ── 1. 기본 설정 및 모델 로드 ──────────────────────────────
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("[1/5] SBERT 모델 로드 중...")
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

target_concept = "컨텍 우주 위성 지상국 발사체 우주산업 딥테크 안테나"
target_emb = model.encode([target_concept])[0]

# ── 2. Semantic Restorer (초강력 정제 & Centrality) ────────
class SemanticRestorer:
    def truncate_tail(self, text):
        if not isinstance(text, str): return ""
        tail_markers = r'(랭킹뉴스|기자의\s*다른\s*기사|회사소개|모바일버전|전체메뉴|ⓒ|무단\s*전재|Copyright|다른기사\s*보기)'
        parts = re.split(tail_markers, text, flags=re.IGNORECASE)
        return parts[0].strip()

    def hard_drop_check(self, text):
        drop_keywords = ['랭킹뉴스', '전체메뉴', '회사소개', '모바일버전', '공유하기']
        if any(k in text for k in drop_keywords): return True
        categories = ['정치', '행정', '경제', '사회', '문화', '스포츠', '연예', '부동산', '운세']
        if sum(1 for c in categories if c in text) >= 4: return True
        return False

    def clean_boilerplate(self, text):
        cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', text)
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!]', ' ', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def calculate_centrality(self, title, text):
        """[지시 반영] Centrality 고도화 (위치, 밀도, 나열형 페널티)"""
        score = 0.0
        title_str = str(title).lower().strip()
        text_str = str(text).lower().strip()
        
        # 1. 제목 가중치
        if title_str.startswith('컨텍') or title_str.startswith('contec'):
            score += 0.5
        elif '컨텍' in title_str or 'contec' in title_str:
            score += 0.3
            
        # 2. 본문 초반 등장 (First 15%)
        first_idx = text_str.find('컨텍')
        if first_idx == -1: first_idx = text_str.find('contec')
        
        if first_idx != -1:
            rel_pos = first_idx / max(len(text_str), 1)
            if rel_pos <= 0.15:
                score += 0.3
                
        # 3. 등장 밀도 가중치 (Density)
        words = text_str.split()
        count = text_str.count('컨텍') + text_str.count('contec')
        density = count / max(len(words), 1)
        score += min(density * 10, 0.4) # 최대 0.4점
        
        # 4. 단순 나열형 리스트 페널티 (A, B, 컨텍, C)
        if re.search(r'(,\s*컨텍|컨텍\s*,|,\s*contec|contec\s*,)', text_str) and count <= 2:
            score -= 0.5
            
        return score

# ── 3. 파이프라인 1단계: 정제 및 초기 스코어링 ──────────────
print("[2/5] 정제 및 Centrality 계산 중...")
df = pd.read_csv("../data/raw_articles.csv")
restorer = SemanticRestorer()

df['raw_body'] = df.apply(lambda row: row['content'] if pd.notna(row.get('content')) else row['snippet'], axis=1)

valid_indices = []
for idx, row in df.iterrows():
    title = str(row['title'])
    body = restorer.truncate_tail(str(row['raw_body']))
    
    if restorer.hard_drop_check(body): continue
    clean_body = restorer.clean_boilerplate(body)
    if len(clean_body) < 50: continue
        
    centrality = restorer.calculate_centrality(title, clean_body)
    
    # 임베딩 오염 방지를 위해 최소 Centrality 0 이상만 통과
    if centrality >= 0.0:
        valid_indices.append(idx)
        df.at[idx, 'purified_body'] = clean_body
        df.at[idx, 'centrality'] = centrality
        
filtered_df = df.loc[valid_indices].copy().reset_index(drop=True)

# ── 4. 파이프라인 2단계: Duplicate Collapse ─────────────────
print(f"[3/5] Duplicate Collapse (유사도 0.90 기준) 진행 중... (현재 {len(filtered_df)}건)")
if len(filtered_df) > 0:
    doc_embeddings = model.encode(filtered_df['purified_body'].tolist())
    
    # 1 - 코사인 유사도로 거리 매트릭스 생성
    dist_matrix = 1 - cosine_similarity(doc_embeddings)
    np.fill_diagonal(dist_matrix, 0)
    
    # 유사도 0.90(거리 0.1) 이상인 기사 묶기
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.1)
    filtered_df['cluster_id'] = clustering.fit_predict(dist_matrix)
    
    # 클러스터별로 Centrality가 가장 높은 대표 기사 1건만 유지
    collapsed_df = filtered_df.sort_values('centrality', ascending=False).drop_duplicates(subset=['cluster_id'], keep='first').copy()
    collapsed_df = collapsed_df.reset_index(drop=True)
else:
    collapsed_df = filtered_df.copy()

print(f"✔️ 중복 제거 완료: {len(collapsed_df)}건의 유니크한 중심 담론 확보")

# ── 5. 파이프라인 3단계: Score Normalization ────────────────
print("[4/5] Score Normalization (변별력 확보) 중...")
if len(collapsed_df) > 0:
    final_embeddings = model.encode(collapsed_df['purified_body'].tolist())
    raw_sims = cosine_similarity(final_embeddings, [target_emb]).flatten()
    
    # MinMax Scaler를 통해 0~1로 분포를 넓게 쫙 펴줌
    scaler = MinMaxScaler()
    norm_sims = scaler.fit_transform(raw_sims.reshape(-1, 1)).flatten()
    
    # 최종 Score = 정규화된 유사도(0.6) + 중심성(0.4)
    collapsed_df['norm_rel_score'] = (norm_sims * 0.6) + (collapsed_df['centrality'] * 0.4)
    
    # 샘플 저장
    collapsed_df[['title', 'purified_body', 'centrality', 'norm_rel_score']].sort_values(by='norm_rel_score', ascending=False).to_csv(OUTPUT_DIR / "phase5_normalized_space.csv", index=False, encoding='utf-8-sig')

# ── 6. 시각화: t-SNE Semantic Geometry ──────────────────────
print("[5/5] UMAP/t-SNE 기반 공간 시각화 중...")
anchors = {
    "Deep Tech (기술)": "위성 데이터 지상국 RF 안테나 우주 통신 EO IR 광통신 SSA 저궤도 기술력 인프라",
    "Theme (테마주)": "수혜주 주가 급등 목표가 거래량 테마주 우주항공청 관련주 변동성 증시 상장",
    "Ecosystem (생태계/노드)": "우주항공청 대전 클러스터 산학연 국가 전략 정책 스타트업 생태계 육성 협력"
}

anchor_embs = model.encode(list(anchors.values()))
anchor_labels = list(anchors.keys())

# 문서 임베딩과 Anchor 임베딩 결합
all_embs = np.vstack([final_embeddings, anchor_embs])

# 데이터 개수가 적을 때를 대비한 perplexity 자동 조절
perplexity_val = min(5, len(all_embs) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
proj_2d = tsne.fit_transform(all_embs)

doc_proj = proj_2d[:-3] # 문서 좌표
anchor_proj = proj_2d[-3:] # Anchor 좌표

plt.figure(figsize=(10, 8))

# 1. 문서 산점도 (점수에 따라 색상 부여)
scatter = plt.scatter(doc_proj[:, 0], doc_proj[:, 1], 
                      c=collapsed_df['norm_rel_score'], cmap='viridis', 
                      s=100, alpha=0.7, edgecolors='w', label='Contec Articles')

# 2. Anchor 별모양 마킹
colors = ['red', 'orange', 'green']
for i, (x, y) in enumerate(anchor_proj):
    plt.scatter(x, y, c=colors[i], marker='*', s=500, edgecolors='black', label=anchor_labels[i])
    plt.text(x, y+0.5, anchor_labels[i], fontsize=12, fontweight='bold', ha='center',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.colorbar(scatter, label='Normalized Relevance Score')
plt.title("Contec Semantic Geometry (t-SNE Projection)", fontsize=16, fontweight='bold')
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

output_img = OUTPUT_DIR / "brand_tsne_geometry.png"
plt.savefig(output_img, dpi=300)
plt.close()

print(f"\n🎉 [실행 완료] t-SNE 시각화가 {output_img.name} 에 저장되었습니다.")