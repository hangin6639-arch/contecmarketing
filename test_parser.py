from bs4 import BeautifulSoup

# 저장된 디버그 HTML 파일 열기
with open("debug_search_page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "lxml")

# 후보 1: 기존 PC 셀렉터
items_pc = soup.select("ul.list_news > li.bx")

# 후보 2: 클래스가 없는 li까지 포함
items_pc_broad = soup.select("ul.list_news > li")

# 후보 3: 모바일 또는 변경된 범용 래퍼
items_mobile = soup.select(".news_wrap, .news_area, .api_ani_send")

print(f"후보 1 개수: {len(items_pc)}")
print(f"후보 2 개수: {len(items_pc_broad)}")
print(f"후보 3 개수: {len(items_mobile)}")

best_items = items_pc or items_pc_broad or items_mobile

if best_items:
    first_item = best_items[0]

    title_tag = first_item.select_one("a.news_tit, .tit, .news_tit")

    title = title_tag.get_text(strip=True) if title_tag else "제목 못 찾음"

    print(f"첫 번째 기사 제목 테스트: {title}")

else:
    print("기사 리스트를 찾지 못했습니다.")