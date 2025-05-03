import os
import math
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from nltk import word_tokenize
import time
from typing import Dict, List, Tuple, Any


def preload_tf_idf_vectors(tf_idf_dir: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    articles_vectors = {}
    idf_dict = defaultdict(list)

    for filename in os.listdir(tf_idf_dir):
        tf_idf_dict = {}
        with open(os.path.join(tf_idf_dir, filename), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                term, idf_val, tf_idf_val = parts[0], float(parts[1]), float(parts[2])
                tf_idf_dict[term] = tf_idf_val

                idf_dict[term].append(idf_val)

        articles_vectors[filename] = tf_idf_dict

    idf = {term: vals[0] for term, vals in idf_dict.items()}
    return articles_vectors, idf


def vectorize_query(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    terms = word_tokenize(query.lower())
    present_terms = [term for term in terms if term in idf]
    tf = Counter(present_terms)
    total = sum(tf.values())
    return {term: (freq / total) * idf.get(term, 0) for term, freq in tf.items() if term in idf}


def calculate_cos_similarity(vector_1: Dict[str, float], vector_2: Dict[str, float]) -> float:
    terms_in_common = set(vector_1.keys()) & set(vector_2.keys())

    if not terms_in_common:
        return 0.0

    scalar_multiply = sum(vector_1[t] * vector_2[t] for t in terms_in_common)

    vector_1_norm = math.sqrt(sum(v ** 2 for v in vector_1.values()))
    vector_2_norm = math.sqrt(sum(v ** 2 for v in vector_2.values()))

    if vector_1_norm and vector_2_norm:
        return scalar_multiply / (vector_1_norm * vector_2_norm)
    else:
        return 0.0


def get_article_title(pages_dir: str, article_id: str) -> str:
    path = os.path.join(pages_dir, article_id)
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.string.strip() if soup.title and soup.title.string else "Без заголовка"


def find_top_articles(
        query: str,
        articles_vectors: Dict[str, Dict[str, float]],
        idf: Dict[str, float],
        pages_dir: str,
        articles_count: int
) -> Tuple[int, List[Dict[str, Any]], float]:

    start_time = time.time()

    query_vector = vectorize_query(query, idf)

    if not query_vector:
        return 0, [], round(time.time() - start_time, 2)

    scores = []
    for article_id, vec in articles_vectors.items():
        if not any(term in vec for term in query_vector):
            continue

        cos_similarity = calculate_cos_similarity(query_vector, vec)

        if cos_similarity > 0:
            scores.append((article_id, cos_similarity))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for article_id, cos_similarity in scores[:articles_count]:
        results.append({
            "article_id": article_id,
            "cos_similarity": cos_similarity,
            "title": get_article_title(pages_dir, article_id)
        })

    elapsed_time = round(time.time() - start_time, 2)
    return len(scores), results, elapsed_time


def launch_searcher_cli(pages_dir: str, tf_idf_dir: str, articles_count: int):
    articles_tf_idf_vectors, idf = preload_tf_idf_vectors(tf_idf_dir)

    while True:
        query_str = input("Введите запрос > ").strip()

        count, results, elapsed_time = find_top_articles(
            query=query_str,
            articles_vectors=articles_tf_idf_vectors,
            idf=idf,
            pages_dir=pages_dir,
            articles_count=articles_count,
        )

        if not results:
            print("\nПо вашему запросу ничего не найдено.\n")
            continue

        print(f"\nВсего результатов: {count}, elapsed_time: {elapsed_time} сек.")
        print(f"\nПервые {articles_count} результатов по вашему запросу:\n")

        for i, article in enumerate(results, 1):
            print(f"{i}. {article['title']}")
            print(f"   {article['article_id']} (cos_similarity: {article['cos_similarity']:.4f})\n")


if __name__ == '__main__':
    top_articles_count = 10
    launch_searcher_cli("../scrapper/saved_pages", "../tf_idf/lemmas", top_articles_count)
