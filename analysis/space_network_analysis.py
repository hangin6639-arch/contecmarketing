"""
한국 우주산업 지식그래프 구축 파이프라인 (Phase 6: Network Intelligence)
========================================================================================
목표: 잔여 UI 노이즈 완벽 제거 후, 기업-기술-정책 간의 Co-occurrence Graph를 생성하여 매개 중심성(Betweenness)을 분석.
"""

import pandas as pd
import numpy as np
import re
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 1. 기본 설정 ───────────────────────────────────────────
OUTPUT_DIR = Path("../outputs/network")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 2. 우주산업 전용 지식 사전 (Entity Dictionary) ─────────
ENTITY_DICT = {
    # 1. 핵심 기업 (Companies)
    "기업": [
        "컨텍", "CONTEC", "한화에어로스페이스", "쎄트렉아이", "AP위성", 
        "한화시스템", "켄코아에어로스페이스", "켄코아", "제노코", "루미르", 
        "페리지에어로스페이스", "이노스페이스", "한국항공우주", "KAI", "대한항공"
    ],
    # 2. 정책/기관 (Policy & Institutions)
    "정책_기관": [
        "우주항공청", "KASA", "대전시", "제주도", "제주", "KAIST", "카이스트",
        "한국항공우주연구원", "항우연", "국가위성운영센터", "우주산업 클러스터", "국가산단"
    ],
    # 3. 핵심 기술/인프라 (Tech & Infrastructure)
    "기술_섹터": [
        "지상국", "위성 데이터", "위성통신", "발사체", "SAR", "초소형 위성", 
        "뉴스페이스", "New Space", "저궤도", "광통신"
    ]
}

# 검색 속도 향상을 위한 평탄화 및 정규화
ALL_ENTITIES = []
for category, entities in ENTITY_DICT.items():
    ALL_ENTITIES.extend(entities)
ALL_ENTITIES = list(set(ALL_ENTITIES)) # 중복 제거

# ── 3. Data Purification Layer (강력한 UI 노이즈 제거) ─────
class NetworkPurifier:
    def __init__(self):
        # [지시 1단계] 잔여 UI 찌꺼기 완벽 제거 정규식
        self.hardcore_noise_patterns = [
            r'댓글\s*\d*', r'공유하기', r'URL복사', r'로그인\s*후\s*이용.*', 
            r'기사스크랩', r'비밀번호', r'SNS\s*보내기', r'기자소개', 
            r'다른\s*기사\s*보기', r'페이스북\(으\)로.*', r'트위터\(으\)로.*',
            r'카카오스토리\(으\)로.*', r'네이버블로그\(으\)로.*', r'핀터레스트\(으\)로.*',
            r'닫기', r'삭제한\s*댓글은.*', r'그래도\s*삭제하시겠습니까', r'댓글\s*내용입력',
            r'[가-힣]{2,4}\s*기자\s*=', r'무단전재', r'재배포\s*금지', r'ⓒ'
        ]

    def purify_text(self, text):
        if not isinstance(text, str): return ""
        cleaned = text
        for pat in self.hardcore_noise_patterns:
            cleaned = re.sub(pat, ' ', cleaned, flags=re.IGNORECASE)
        # 이메일 및 불필요한 특수문자 제거
        cleaned = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' ', cleaned)
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s.,]', ' ', cleaned)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def extract_entities(self, text):
        """[지시 2단계] 텍스트 내 지식 사전 Entity 추출"""
        found_entities = set()
        text_upper = text.upper()
        for entity in ALL_ENTITIES:
            # 영문은 대소문자 무시, 한글은 그대로 매칭
            if entity.upper() in text_upper:
                # 동의어 통합 (예: 켄코아에어로스페이스 -> 켄코아, KASA -> 우주항공청)
                if entity.upper() in ["CONTEC", "컨텍"]: found_entities.add("컨텍")
                elif entity.upper() in ["KAI", "한국항공우주"]: found_entities.add("한국항공우주")
                elif entity.upper() in ["KASA", "우주항공청"]: found_entities.add("우주항공청")
                elif entity.upper() in ["켄코아에어로스페이스", "켄코아"]: found_entities.add("켄코아")
                elif entity.upper() in ["제주도", "제주"]: found_entities.add("제주도")
                elif entity.upper() in ["카이스트", "KAIST"]: found_entities.add("KAIST")
                else: found_entities.add(entity)
        return list(found_entities)

# ── 4. 파이프라인 실행: 정제 및 Entity 추출 ────────────────
print("[1/4] 데이터 로드 및 Hardcore UI 정제 중...")
df = pd.read_csv("../data/raw_articles.csv")
purifier = NetworkPurifier()

df['raw_body'] = df.apply(lambda row: row['content'] if pd.notna(row.get('content')) else row['snippet'], axis=1)
df['clean_body'] = df['raw_body'].apply(purifier.purify_text)

print("[2/4] Entity Extraction (사전 기반 기업/정책/기술 추출) 중...")
df['entities'] = df['clean_body'].apply(purifier.extract_entities)

# Entity가 2개 이상 존재하는 유의미한 기사만 필터링 (연결망 구축용)
network_df = df[df['entities'].apply(len) >= 2].copy()
print(f"✔️ 네트워크 분석 유효 기사: {len(network_df)}건 확보")

# ── 5. Co-occurrence Graph 생성 ────────────────────────────
print("[3/4] Co-occurrence Graph 생성 및 중심성(Centrality) 분석 중...")
G = nx.Graph()

# [지시 3단계] 문서 단위 Edge 생성
for entities in network_df['entities']:
    # 엔티티 간의 모든 2개 조합 생성
    for u, v in itertools.combinations(entities, 2):
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

# [지시 4단계 & 5단계] 중심성 분석 (매개 중심성 최우선)
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight='weight')
eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)

# 노드 속성으로 저장
nx.set_node_attributes(G, degree_cent, 'degree')
nx.set_node_attributes(G, betweenness_cent, 'betweenness')
nx.set_node_attributes(G, eigenvector_cent, 'eigenvector')

# 결과를 DataFrame으로 변환하여 저장
nodes_data = []
for node in G.nodes():
    nodes_data.append({
        'Node': node,
        'Degree': degree_cent[node],
        'Betweenness (매개)': betweenness_cent[node],
        'Eigenvector (영향력)': eigenvector_cent[node]
    })

node_df = pd.DataFrame(nodes_data).sort_values(by='Betweenness (매개)', ascending=False)
node_df.to_csv(OUTPUT_DIR / "network_centrality_metrics.csv", index=False, encoding='utf-8-sig')

# ── 6. 네트워크 시각화 ─────────────────────────────────────
print("[4/4] 우주산업 지식그래프 시각화 생성 중...")
plt.figure(figsize=(14, 10))

# 노드 크기는 Betweenness Centrality에 비례
node_sizes = [v * 10000 for v in betweenness_cent.values()]
# Edge 굵기는 가중치에 비례
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# 레이아웃 설정 (Spring Layout)
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# 노드 및 엣지 그리기
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3498db', alpha=0.7, edgecolors='white')
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4, edge_color='#95a5a6')

# 노드 라벨 (Betweenness 상위 노드만 폰트 강조)
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_family='Malgun Gothic', font_size=10, font_weight='bold')

plt.title("한국 우주산업 Co-occurrence 지식그래프 (컨텍 브리지 분석)", fontsize=18, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "space_industry_knowledge_graph.png", dpi=300)
plt.close()

print(f"\n🎉 [실행 완료] outputs/network 폴더의 CSV 지표와 네트워크 그래프를 확인하십시오.")
print(f"💡 [핵심 인사이트 분석 대기] 컨텍의 Betweenness(매개 중심성) 순위를 확인해 주십시오!")