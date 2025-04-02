import math
import os
import time
import requests
from bs4 import BeautifulSoup

# Настройки
hub = "kotlin"
base_url_template = f"https://habr.com/ru/hubs/{hub}/articles/top/yearly/page{{}}"
article_url_template = "https://habr.com/ru/articles/{}"

saved_pages_dir = "saved_pages"

articles_per_page = 20
required_articles_count = 160

max_pages = math.ceil(required_articles_count / articles_per_page)

# Задержка между запросами в секундах
request_delay_seconds = 0.5

# Файл индекса
index_file_name = "index.txt"


def start_scraping():
    """Запуск синхронного сканирования и скачивания статей."""
    already_loaded_or_visited, last_index = load_visited_articles(index_file_name)
    visited_count = 0
    saved_count = 0

    # Обход страниц /pageN/
    for page_number in range(1, max_pages + 1):
        page_url = base_url_template.format(page_number)
        print(f"Сканируется {page_url}")

        article_ids = extract_article_ids(page_url)
        for article_id in article_ids:
            article_url = article_url_template.format(article_id)

            if article_url in already_loaded_or_visited:
                continue

            article_html = get_page(article_url)
            visited_count += 1

            already_loaded_or_visited.add(article_url)

            if article_html:
                saved_count += 1
                last_index += 1
                save_page(last_index, article_id, article_html)

            time.sleep(request_delay_seconds)

    print(f"Загрузка завершена: посещено {visited_count} страниц, скачано {saved_count} статей. Всего статей в индексе: {last_index}")


def get_page(url: str):
    """Загрузка страницы."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[x] Ошибка при скачивании {url}: {e}")
        return None


def extract_article_ids(page_url: str):
    """Извлекает ID статей из списка статей на странице."""
    html = get_page(page_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    articles_list = soup.find("div", class_="tm-articles-list")

    if not articles_list:
        print(f"Не найден список статей на странице {page_url}")
        return []

    article_ids = []
    for article in articles_list.find_all("article", class_="tm-articles-list__item"):
        article_id = article.get("id")
        article_ids.append(article_id)

    return article_ids


def remove_tags(text: str, tags: list) -> str:
    """Очищение HTML от выбранных тегов"""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(tags):
        tag.decompose()
    return str(BeautifulSoup(str(soup), 'html.parser'))


def save_page(index: int, article_id: str, html: str):
    """Сохраняет страницу и обновляет index.txt."""
    file_name = f"article_{article_id}.txt"
    file_path = os.path.join(saved_pages_dir, file_name)

    cleaned_html = remove_tags(html, ["script", "style", "link", "meta"])

    with open(file_path, "w", encoding="utf-8") as page_file:
        page_file.write(cleaned_html)

    with open(index_file_name, "a", encoding="utf-8") as file:
        file.write(f"{index}: {article_url_template.format(article_id)}\n")

    print(f"[✓] Статья {article_id} сохранена.")


def load_visited_articles(index_file: str):
    """Загружает список уже скачанных статей из index.txt."""
    visited = set()
    max_index = 0
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(": ")
                visited.add(parts[1])
                max_index = parts[0]

    return visited, int(max_index)


if __name__ == "__main__":
    os.makedirs(saved_pages_dir, exist_ok=True)
    start_scraping()
