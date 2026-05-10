"""
Contec Identity Corpus Builder (Phase 7-A)
======================================================================
목표: 홈페이지 및 사업보고서(PDF)에서 텍스트를 추출하여 
Layer 1(Identity)과 Layer 2(Curated Image)의 Semantic Corpus 구축.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 1. 기본 설정 ───────────────────────────────────────────
OUTPUT_DIR = Path("../outputs/identity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 2. 추출 및 정제 클래스 ────────────────────────────────
class CorporateDataExtractor:
    def __init__(self):
        self.corpus_data = []

    def clean_corporate_text(self, text):
        """기업 텍스트 특화 정제 (Boilerplate 및 UI 노이즈 제거)"""
        if not isinstance(text, str): return ""
        # HTML 태그, 과도한 공백, 홈페이지 Footer 정보 제거
        cleaned = re.sub(r'<[^>]+>', ' ', text)
        cleaned = re.sub(r'(Copyright|ⓒ|All Rights Reserved|개인정보처리방침|이용약관).*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def split_into_sentences(self, text):
        """문장 단위로 분할하여 SBERT 임베딩 효율 극대화"""
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20] # 너무 짧은 메뉴명 제외

    def scrape_website(self, url, source_name, layer):
        """웹페이지 텍스트 크롤링"""
        print(f"🌐 [웹 크롤링] {source_name} 추출 중...")
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # script, style 태그 제거
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
                
            raw_text = soup.get_text(separator=' ')
            clean_text = self.clean_corporate_text(raw_text)
            sentences = self.split_into_sentences(clean_text)
            
            for s in sentences:
                self.corpus_data.append({"layer": layer, "source": source_name, "sentence": s})
        except Exception as e:
            print(f"웹 크롤링 오류 ({url}): {e}")

    def extract_from_pdf(self, pdf_path, source_name, layer):
        """사업보고서/IR 등 PDF 텍스트 추출"""
        print(f"📄 [PDF 파싱] {source_name} 추출 중...")
        try:
            doc = fitz.open(pdf_path)
            raw_text = ""
            for page in doc:
                raw_text += page.get_text("text") + " "
            
            clean_text = self.clean_corporate_text(raw_text)
            sentences = self.split_into_sentences(clean_text)
            
            for s in sentences:
                # '사업의 내용', '당사는' 등의 1인칭 관점 문장 위주로 수집
                if any(kw in s for kw in ['당사는', '컨텍은', '제공', '사업', '인프라', '솔루션']):
                    self.corpus_data.append({"layer": layer, "source": source_name, "sentence": s})
        except Exception as e:
            print(f"PDF 파싱 오류 ({pdf_path}): {e}")

# ── 3. 코퍼스 구축 실행 ──────────────────────────────────
if __name__ == "__main__":
    extractor = CorporateDataExtractor()

    # [Layer 1] Internal Identity 구축 (기업 소개 & 사업보고서)
    print("\n[🚀 Layer 1: Internal Identity 데이터 구축 시작]")
    extractor.scrape_website("https://kr.contec.kr/gwbbs/gw_page15.php?ti=cate1&me_co=1010", "Homepage_About", "Layer_1_Identity")
    # 주의: PDF 파일은 파이썬 파일이 있는 곳과 동일한 경로나 지정된 경로에 있어야 합니다.
    pdf_file_path = "../data/[컨텍]사업보고서(2026.03.23).pdf" 
    if Path(pdf_file_path).exists():
        extractor.extract_from_pdf(pdf_file_path, "Business_Report_2026", "Layer_1_Identity")
    else:
        print(f"⚠️ PDF 파일을 찾을 수 없습니다. 경로를 확인해 주세요: {pdf_file_path}")

    # [Layer 2] Curated External Image 구축 (뉴스룸/PR 보도자료)
    print("\n[🚀 Layer 2: Curated External Image 데이터 구축 시작]")
    extractor.scrape_website("https://kr.contec.kr/gwbbs/board.php?bo_table=bbs4&me_co=2010", "Homepage_PR_News", "Layer_2_Curated_Image")

    # DataFrame 변환 및 저장
    corpus_df = pd.DataFrame(extractor.corpus_data)
    
    # 중복 문장 제거
    corpus_df = corpus_df.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    
    save_path = OUTPUT_DIR / "contec_brand_corpus.csv"
    corpus_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 [실행 완료] {len(corpus_df)}개의 공식 브랜드 문장(Ground Truth)이 추출되었습니다.")
    print(f"결과물 저장 위치: {save_path}")
    print("\n이 코퍼스는 Phase 7-B(Semantic Gap 분석)의 절대적 기준점(Anchor)으로 사용됩니다.")