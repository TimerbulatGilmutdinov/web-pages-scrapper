from pathlib import Path
from collections import defaultdict


def make_index(lemmas_path: str) -> dict[str, set[int]]:
    index = defaultdict(set[int])
    lemmas_dir = Path(lemmas_path)

    for file in lemmas_dir.iterdir():
        name_parts = file.stem.split("_")
        file_number = int(name_parts[-1])

        with file.open(encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    index[tokens[0]].add(file_number)
    return index


def write_index_file(index_data: dict[str, set[int]], output_file: str):
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("lemma, articles\n")
        for term in sorted(index_data):
            ids_str = " ".join(str(i) for i in sorted(index_data[term]))
            out.write(f"{term}, {ids_str}\n")


def read_index_file(file_name: str) -> dict[str, set[int]]:
    index = defaultdict(set[int])
    with open(file_name, "r", encoding="utf-8") as file:
        next(file)
        for line in file:
            line = line.strip()
            lemma, article_ids = line.split(", ")
            articles_set = set(map(int, article_ids.split()))
            index[lemma] = articles_set
    return index


def main():
    source_dir = "../lemmatizer/lemmas"
    output_file = "inverted_index.csv"
    index = make_index(source_dir)
    write_index_file(index, output_file)


if __name__ == "__main__":
    main()
