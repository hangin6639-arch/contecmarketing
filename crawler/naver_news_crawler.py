"""
네이버 뉴스 크롤러 — 컨텍(Contec) 브랜드 담론 분석용
=======================================================
수집 대상 : 네이버 뉴스 검색 결과 (비공식 크롤링)
기간      : 2023-01-01 ~ 2024-12-31
출력      : data/raw_articles.csv

사용법:
    python naver_news_crawler.py
    python naver_news_crawler.py --query "컨텍" --start 2023-01-01 --end 2024-12-31
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from playwright.sync_api import sync_playwright

try:
    from newspaper import Article as NewspaperArticle
    _NEWSPAPER_AVAILABLE = True
except ImportError:
    _NEWSPAPER_AVAILABLE = False

# ── 정제 패턴 (기자명 / 광고 / 노이즈) ────────────────────────────────────────
_BYLINE_PATTERN = re.compile(
    r"""
    (?:
        [\(\[\<]?\s*[가-힣]{2,5}\s*[\)\]\>]?\s*(?:기자|특파원|에디터|리포터|선임|취재|편집)\s*
        | ^\s*기자\s*[=:]\s*[가-힣]{2,5}\s*$
        | \[\s*[가-힣]{2,5}\s*기자\s*\]
        | ©\s*\S+
        | ▶\s*.*?$
        | \【.*?\】
        | 무단\s*전재.*?금지
        | 저작권.*?금지
        | 뉴스\s*제보.*?$
        | 구독.*?클릭
        | 광고\s*문의.*?$
    )
    """,
    re.VERBOSE | re.MULTILINE,
)

_AD_LINE_PATTERNS = re.compile(
    r"^.{0,30}(?:"
    r"구독하기|뉴스레터|앱\s*다운|QR\s*코드|제보하기|"
    r"카카오톡|텔레그램|유튜브\s*채널|인스타그램|"
    r"더보기\s*▶|관련\s*기사|포토뉴스|영상뉴스|"
    r"Copyright\s*©|All\s*rights\s*reserved"
    r").{0,60}$",
    re.MULTILINE | re.IGNORECASE,
)

# ── 로깅 설정 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crawler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────
DEFAULT_QUERY   = "컨텍 우주"
DEFAULT_START   = "2023-01-01"
DEFAULT_END     = "2024-12-31"
DELAY_MIN       = 1.5
DELAY_MAX       = 3.5
MAX_PAGES       = 40
OUTPUT_DIR      = Path("data")
OUTPUT_FILE     = OUTPUT_DIR / "raw_articles.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://news.naver.com/",
}

# ── 핵심 함수 ─────────────────────────────────────────────────────────────────

def build_search_url(query: str, page: int, start_date: str, end_date: str) -> str:
    """네이버 뉴스 검색 URL 생성."""
    ds = start_date.replace("-", "")
    de = end_date.replace("-", "")
    start_idx = (page - 1) * 10 + 1
    return (
        f"https://search.naver.com/search.naver"
        f"?where=news&query={requests.utils.quote(query)}"
        f"&nso=so:dd,p:from{ds}to{de}"
        f"&start={start_idx}"
    )

def fetch_search_page_playwright(url: str, page_obj) -> BeautifulSoup | None:
    """Playwright를 이용한 검색 결과 페이지 요청 (Timeout 우회 버전)"""
    try:
        # networkidle 대신 domcontentloaded로 변경하여 타임아웃 방지 (15초 제한)
        page_obj.goto(url, wait_until="domcontentloaded", timeout=15000)
        
        # DOM 로드 직후 동적 렌더링(기사 카드 생성)을 기다리기 위한 짧은 물리적 대기
        page_obj.wait_for_timeout(2500) 
        
        html = page_obj.content()
        return BeautifulSoup(html, "lxml")
    except Exception as e:
        log.warning(f"Playwright 페이지 요청 실패: {e}")
        return None

def parse_article_list(soup: BeautifulSoup) -> list[dict]:
    """최신 네이버 뉴스 UI(SDS/Fender 컴포넌트) 완벽 대응 파서"""
    articles = []

    # 1. 신규 UI 기사 컨테이너 타겟팅 (CSS 속성 선택자 사용)
    # 구형 UI(.list_news > li.bx)도 하위 호환으로 남겨둠
    items = soup.select("ul.list_news > li.bx, ul.list_news > li")
    if not items:
        # class 이름에 'sds-comps-vertical-layout'가 포함된 모든 div 탐색
        items = soup.select("div[class*='sds-comps-vertical-layout']")

    for item in items:
        try:
            # 1. 제목 및 URL 추출
            # 해당 컨테이너 안의 모든 a 태그를 가져옴
            anchors = item.find_all("a", href=True)
            
            # [핵심] 네이버 헬프, 더미 링크, 자동완성 노이즈 필터링
            valid_anchors = [
                a for a in anchors 
                if "help.naver.com" not in a['href'] 
                and a['href'] != "#"
                and "자동완성" not in a.get_text(strip=True)
            ]

            if not valid_anchors:
                continue

            # 유효한 a 태그 중 텍스트가 가장 긴 것을 '기사 제목'으로 간주
            title_tag = max(valid_anchors, key=lambda a: len(a.get_text(strip=True)))
            title = title_tag.get_text(strip=True)
            url = title_tag['href']

            # 텍스트가 너무 짧으면 기사가 아니라고 판단
            if len(title) < 8:
                continue

            # 2. 언론사 추출 (보통 2~10자 사이의 짧은 링크 텍스트)
            source = "알수없음"
            source_candidates = [
                a for a in valid_anchors 
                if 2 <= len(a.get_text(strip=True)) <= 12 and a != title_tag
            ]
            if source_candidates:
                source = source_candidates[0].get_text(strip=True).replace("언론사 선정", "").strip()

            # 3. 날짜 추출
            date_str = ""
            all_texts = item.find_all(["span", "div", "em"])
            for tag in all_texts:
                text = tag.get_text(strip=True)
                # 상대 날짜 패턴
                if any(kw in text for kw in ["시간 전", "분 전", "일 전", "주 전", "어제", "오늘"]):
                    date_str = text
                    break
                # 절대 날짜 패턴 (예: "2024.05.08." 또는 "2024.05.08")
                elif re.match(r"^\d{4}\.\d{2}\.\d{2}\.?$", text):
                    date_str = text
                    break

            # 4. 요약문 추출
            # 컨테이너 안의 div 중 텍스트가 가장 긴 것을 요약문으로 간주
            snippet = ""
            divs = item.find_all("div")
            valid_divs = [d for d in divs if d.get_text(strip=True) and d.get_text(strip=True) != title]
            if valid_divs:
                desc_tag = max(valid_divs, key=lambda d: len(d.get_text(strip=True)))
                # 제목이나 날짜가 중복으로 딸려오는 것 방지
                snippet = desc_tag.get_text(strip=True)
                snippet = snippet.replace(title, "").replace(date_str, "").strip()

            articles.append({
                "title":   title,
                "url":     url,
                "source":  source,
                "date_raw": date_str,
                "snippet": snippet[:300], # 너무 긴 텍스트는 잘라냄
            })
        except Exception as e:
            continue

    # 5. 중복 기사 제거 (동일 URL 기준)
    unique_articles = []
    seen_urls = set()
    for art in articles:
        if art['url'] not in seen_urls:
            unique_articles.append(art)
            seen_urls.add(art['url'])

    return unique_articles

def _clean_body(text: str) -> str:
    """공통 본문 후처리."""
    text = _BYLINE_PATTERN.sub("", text)
    text = _AD_LINE_PATTERNS.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# ── 본문 추출 전략 (우선순위 순) ──────────────────────────────────────────────
MIN_BODY_LEN = 150

_NAVER_SELECTORS = [
    "div#newsct_article",
    "div.go_trans._article_content",
    "div#articeBody",
    "div#articleBodyContents",
    "div._article_body",
]

_GENERIC_SELECTORS = [
    "article",
    "div.article-body", "div.article_body", "div.articleBody",
    "div.article-content", "div.article_content",
    "div#article-body", "div#articleBody",
    "div.news-content", "div.news_content",
    "div.story-body", "div.story_body",
    "section.article-section",
    "div.view_text",
    "div#newsEndContents",
    "div.read_body",
    "div#content-body",
]

def _extract_with_newspaper(url: str) -> tuple[str, str]:
    if not _NEWSPAPER_AVAILABLE:
        return "", ""
    try:
        art = NewspaperArticle(url, language="ko")
        art.download()
        art.parse()
        text = art.text or ""
        if len(text) >= MIN_BODY_LEN:
            return text, "newspaper3k"
    except Exception as e:
        log.debug(f"[newspaper3k] 실패 ({url}): {e}")
    return "", ""

def _extract_with_naver_selectors(soup: BeautifulSoup) -> tuple[str, str]:
    for selector in _NAVER_SELECTORS:
        tag = soup.select_one(selector)
        if tag:
            for noise in tag.select("script, style, figure, .reporter_area, .copyright"):
                noise.decompose()
            text = tag.get_text(separator="\n", strip=True)
            if len(text) >= MIN_BODY_LEN:
                return text, f"bs4:naver({selector})"
    return "", ""

def _extract_with_generic_selectors(soup: BeautifulSoup) -> tuple[str, str]:
    for selector in _GENERIC_SELECTORS:
        tag = soup.select_one(selector)
        if tag:
            for noise in tag.select("script, style, figure, aside, nav, .ad, .advertisement"):
                noise.decompose()
            text = tag.get_text(separator="\n", strip=True)
            if len(text) >= MIN_BODY_LEN:
                return text, f"bs4:generic({selector})"
    return "", ""

def _extract_body_text(soup: BeautifulSoup) -> tuple[str, str]:
    body = soup.find("body")
    if body:
        for noise in body.select("script, style, header, footer, nav, aside"):
            noise.decompose()
        text = body.get_text(separator="\n", strip=True)
        if len(text) >= 50:
            return text, "bs4:body_fallback"
    return "", ""

def fetch_article_body(url: str, session: requests.Session) -> dict:
    """계층형 기사 본문 추출 엔진."""
    result = {"content": "", "extract_method": "failed", "content_len": 0}

    text, method = _extract_with_newspaper(url)
    if text:
        result["content"]        = _clean_body(text)
        result["extract_method"] = method
        result["content_len"]    = len(result["content"])
        log.debug(f"[L1:newspaper3k] {len(result['content'])}자 ({url[:60]})")
        return result

    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
    except Exception as e:
        log.debug(f"HTML 요청 실패 ({url}): {e}")
        return result

    text, method = _extract_with_naver_selectors(soup)
    if text:
        result["content"]        = _clean_body(text)
        result["extract_method"] = method
        result["content_len"]    = len(result["content"])
        log.debug(f"[L2:naver] {len(result['content'])}자 ({url[:60]})")
        return result

    text, method = _extract_with_generic_selectors(soup)
    if text:
        result["content"]        = _clean_body(text)
        result["extract_method"] = method
        result["content_len"]    = len(result["content"])
        log.debug(f"[L3:generic] {len(result['content'])}자 ({url[:60]})")
        return result

    text, method = _extract_body_text(soup)
    if text:
        result["content"]        = _clean_body(text)
        result["extract_method"] = method
        result["content_len"]    = len(result["content"])
        log.debug(f"[L4:body] {len(result['content'])}자 ({url[:60]})")
        return result

    log.debug(f"[모든 Layer 실패] ({url[:60]})")
    return result

def normalize_date(date_raw: str) -> str:
    now = datetime.now()
    d = date_raw.strip()

    if "분 전" in d or "시간 전" in d or "방금" in d:
        return now.strftime("%Y-%m-%d")
    if "일 전" in d:
        try:
            days = int(d.replace("일 전", "").strip())
            return (now - timedelta(days=days)).strftime("%Y-%m-%d")
        except ValueError:
            pass
    
    clean = d.rstrip(".")
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return d

def is_within_range(date_str: str, start: str, end: str) -> bool:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        return s <= d <= e
    except ValueError:
        return True

# ── 메인 크롤링 루프 ──────────────────────────────────────────────────────────

def crawl(
    query: str = DEFAULT_QUERY,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    max_pages: int = MAX_PAGES,
    fetch_body: bool = True,
) -> pd.DataFrame:
    
    # 본문 수집용 requests 세션 (기존 유지)
    session_req = requests.Session()
    all_rows = []
    seen_urls = set()

    log.info(f"크롤링 시작 | 검색어: '{query}' | 기간: {start_date} ~ {end_date}")

    # 네이버 검색 결과 리스트용 Playwright 컨텍스트
    with sync_playwright() as p:
        # headless=True: 창 숨김 / False: 디버깅용 창 띄움
        browser = p.chromium.launch(
    headless=False,
    slow_mo=50
)
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            viewport={'width': 1920, 'height': 1080}
        )
        page_obj = context.new_page()

        for page in range(1, max_pages + 1):
            url  = build_search_url(query, page, start_date, end_date)
            
            # Playwright로 검색 페이지 HTML 가져오기
            soup = fetch_search_page_playwright(url, page_obj)

            if soup is None:
                log.warning(f"페이지 {page} 수집 실패, 중단")
                break

            articles = parse_article_list(soup)

            if not articles:
                log.info(f"페이지 {page}: 결과 없음 → 크롤링 종료")
                break

            new_count = 0
            for art in articles:
                if art["url"] in seen_urls:
                    continue
                seen_urls.add(art["url"])

                art["date"] = normalize_date(art["date_raw"])

                if not is_within_range(art["date"], start_date, end_date):
                    continue

                if fetch_body:
                    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                    # 본문은 기존 requests 기반 4-Layer 엔진 사용
                    body_result          = fetch_article_body(art["url"], session_req)
                    art["content"]       = body_result["content"]
                    art["extract_method"]= body_result["extract_method"]
                    art["content_len"]   = body_result["content_len"]
                else:
                    art["content"]       = ""
                    art["extract_method"]= "skipped"
                    art["content_len"]   = 0

                all_rows.append(art)
                new_count += 1

            log.info(f"페이지 {page:>3} | 신규 {new_count}건 수집 | 누적 {len(all_rows)}건")
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        browser.close()

    if not all_rows:
        log.warning("수집된 기사가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    cols = ["date", "title", "source", "snippet", "content",
            "extract_method", "content_len", "url", "date_raw"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    if "extract_method" in df.columns:
        log.info("추출 방법 분포:\n" + df["extract_method"].value_counts().to_string())

    log.info(f"크롤링 완료 | 총 {len(df)}건")
    return df

def save(df: pd.DataFrame, path: Path = OUTPUT_FILE) -> None:
    """CSV 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"저장 완료 → {path}")

# ── CLI 진입점 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="네이버 뉴스 크롤러 — 브랜드 담론 분석용")
    parser.add_argument("--query",      default=DEFAULT_QUERY,  help="검색어 (기본: '컨텍 우주')")
    parser.add_argument("--start",      default=DEFAULT_START,  help="시작일 YYYY-MM-DD")
    parser.add_argument("--end",        default=DEFAULT_END,    help="종료일 YYYY-MM-DD")
    parser.add_argument("--pages",      type=int, default=MAX_PAGES, help="최대 페이지 수")
    parser.add_argument("--no-body",    action="store_true",    help="본문 수집 생략 (빠름)")
    parser.add_argument("--output",     default=str(OUTPUT_FILE), help="출력 파일 경로")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    df = crawl(
        query      = args.query,
        start_date = args.start,
        end_date   = args.end,
        max_pages  = args.pages,
        fetch_body = not args.no_body,
    )

    if not df.empty:
        save(df, Path(args.output))
        print("\n── 수집 결과 미리보기 ──────────────────────────────")
        preview_cols = [c for c in ["id", "date", "source", "title", "extract_method", "content_len"] if c in df.columns]
        print(df[preview_cols].head(10).to_string(index=False))
        if "extract_method" in df.columns:
            print(f"\n추출 방법 분포:\n{df['extract_method'].value_counts().to_string()}")
        print(f"\n총 {len(df)}건 → {args.output}")