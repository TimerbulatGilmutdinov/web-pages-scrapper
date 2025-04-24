import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def load_token_data(token_dir: Path) -> Tuple[Dict[str, Counter[str]], Dict[str, int]]:
    doc_token_counts = {}
    token_df = defaultdict(int)

    for file in token_dir.iterdir():
        with file.open('r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
            counter = Counter(tokens)
            doc_token_counts[file.name] = counter
            for token in set(counter):
                token_df[token] += 1

    return doc_token_counts, token_df


def load_lemma_data(
    lemma_dir: Path,
    doc_token_counts: Dict[str, Counter[str]]
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Set[str]]]:
    doc_lemma_tokens = {}
    lemma_doc_set = defaultdict(set)

    for file in lemma_dir.iterdir():
        lemma_map = defaultdict(list)
        with file.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                lemma = parts[0]
                forms = parts[1:]
                lemma_map[lemma].extend(forms)
                if any(t in doc_token_counts[file.name] for t in forms):
                    lemma_doc_set[lemma].add(file.name)
        doc_lemma_tokens[file.name] = lemma_map

    return doc_lemma_tokens, lemma_doc_set


def compute_and_write_tf_idf_tokens(
    doc_token_counts: Dict[str, Counter[str]],
    token_df: Dict[str, int],
    total_docs: int,
    output_dir: Path
) -> None:
    for fname, tokens in doc_token_counts.items():
        lines = []
        total = sum(tokens.values())
        for token, freq in tokens.items():
            tf = freq / total
            idf = math.log(total_docs / token_df[token])
            tf_idf = tf * idf
            lines.append(f"{token} {idf:.6f} {tf_idf:.6f}\n")
        output_dir.joinpath(fname).write_text(''.join(lines), encoding='utf-8')


def compute_and_write_tf_idf_lemmas(
    doc_token_counts: Dict[str, Counter[str]],
    doc_lemma_tokens: Dict[str, Dict[str, List[str]]],
    lemma_doc_set: Dict[str, Set[str]],
    total_docs: int,
    output_dir: Path
) -> None:
    for fname, tokens in doc_token_counts.items():
        lines = []
        total = sum(tokens.values())
        for lemma, forms in doc_lemma_tokens[fname].items():
            freq = sum(tokens.get(t, 0) for t in forms)
            if freq == 0:
                continue
            tf = freq / total
            idf = math.log(total_docs / len(lemma_doc_set[lemma]))
            tf_idf = tf * idf
            lines.append(f"{lemma} {idf:.6f} {tf_idf:.6f}\n")
        output_dir.joinpath(fname).write_text(''.join(lines), encoding='utf-8')


if __name__ == "__main__":
    PAGES_DIR = Path('../scrapper/saved_pages')
    TOKENS_DIR = Path('../lemmatizer/tokens')
    LEMMAS_DIR = Path('../lemmatizer/lemmas')
    OUTPUT_TOKEN_DIR = Path('tokens')
    OUTPUT_LEMMA_DIR = Path('lemmas')

    OUTPUT_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LEMMA_DIR.mkdir(parents=True, exist_ok=True)

    files = {file.name for file in TOKENS_DIR.iterdir()}

    doc_token_counts, token_df = load_token_data(TOKENS_DIR)
    doc_lemma_tokens, lemma_doc_set = load_lemma_data(LEMMAS_DIR, doc_token_counts)

    compute_and_write_tf_idf_tokens(doc_token_counts, token_df, len(files), OUTPUT_TOKEN_DIR)
    compute_and_write_tf_idf_lemmas(doc_token_counts, doc_lemma_tokens, lemma_doc_set, len(files), OUTPUT_LEMMA_DIR)
