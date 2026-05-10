from bs4 import BeautifulSoup
import json

with open("debug_search_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

print("=== 🚀 네이버 신규 UI 셀렉터 탐색 시작 ===")

# 1. 뉴스 링크로 추정되는 모든 a 태그 수집
all_links = soup.find_all("a", href=True)
news_candidates = []

for a in all_links:
    text = a.get_text(strip=True)
    href = a['href']
    
    # 제목에 '컨텍'이나 '우주'가 들어가거나, 텍스트가 15자 이상인 유의미한 텍스트일 경우
    if len(text) > 15 and ("우주" in text or "컨텍" in text):
        # 해당 a 태그의 클래스 목록 확인
        classes = a.get('class', [])
        
        # 부모 컨테이너(보통 기사 1개를 감싸는 카드) 추적
        parent = a.find_parent("div")
        parent_classes = parent.get('class', []) if parent else []
        
        news_candidates.append({
            "title": text,
            "url": href,
            "a_tag_classes": classes,
            "parent_classes": parent_classes
        })

# 결과 출력 (상위 3개만)
for idx, item in enumerate(news_candidates[:3]):
    print(f"\n[기사 {idx+1}]")
    print(f"제목: {item['title']}")
    print(f"URL: {item['url']}")
    print(f"A 태그 클래스: {item['a_tag_classes']}")
    print(f"부모 DIV 클래스: {item['parent_classes']}")

print("\n=> 💡 위 출력 결과에서 'A 태그 클래스'와 '부모 DIV 클래스'를 확인해주세요!")