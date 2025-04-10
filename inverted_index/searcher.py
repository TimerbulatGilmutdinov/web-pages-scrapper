import re
from collections import deque

from nltk.stem import WordNetLemmatizer

from inverted_index.indexer import read_index_file

lemmatizer = WordNetLemmatizer()


def tokenize_query(query: str) -> list[str]:
    regex = r'\(|\)|and|or|not|[a-zA-Z]+'
    tokens = re.findall(regex, query, flags=re.IGNORECASE)

    parsed_tokens = []
    for token in tokens:
        if token.upper() in ["AND", "OR", "NOT"]:
            parsed_tokens.append(token.upper())
        else:
            lemma = lemmatizer.lemmatize(token.lower())
            parsed_tokens.append(lemma)
    print(f"Parsed tokens {parsed_tokens}")
    return parsed_tokens


def process_tokens(input_sequence: list[str]) -> list[str]:
    priority = {"NOT": 3, "AND": 2, "OR": 1}
    output_queue = []
    operators = []

    for symbol in input_sequence:
        if symbol == "(":
            operators.append(symbol)
        elif symbol == ")":
            while operators and operators[-1] != "(":
                output_queue.append(operators.pop())
            if operators:
                operators.pop()
        elif symbol in priority:
            while (
                operators and operators[-1] in priority and
                priority[operators[-1]] >= priority[symbol]
            ):
                output_queue.append(operators.pop())
            operators.append(symbol)
        else:
            output_queue.append(symbol)

    for op in reversed(operators):
        output_queue.append(op)

    return output_queue


def resolve_operator(op: str, lhs: set[int], rhs: set[int] = None, full: set[int] = None) -> set[int]:
    if op == "AND":
        return lhs & rhs
    elif op == "OR":
        return lhs | rhs
    elif op == "NOT":
        return full - lhs
    else:
        return set()


def process_rpn_sequence(sequence: list[str], index: dict[str, set[int]], all_docs: set[int]) -> set[str]:
    buffer = deque()

    for item in sequence:
        if item in {"AND", "OR"}:
            right = buffer.pop()
            left = buffer.pop()
            result = resolve_operator(item, left, right)
            buffer.append(result)
        elif item == "NOT":
            operand = buffer.pop()
            result = resolve_operator("NOT", operand, full=all_docs)
            buffer.append(result)
        else:
            buffer.append(index.get(item, set()))

    return buffer[-1] if buffer else set()

def boolean_search(query: str, inverted_index: dict[str, set[int]]) -> list[str]:
    tokens = tokenize_query(query)
    postfix = process_tokens(tokens)
    article_ids = set()

    for index_article_ids in inverted_index.values():
        article_ids.update(index_article_ids)

    result = process_rpn_sequence(postfix, inverted_index, article_ids)
    return sorted(result)


def main():
    inverted_index = read_index_file("inverted_index.csv")

    while True:
        query = input("search query > ").strip()

        try:
            result_articles = boolean_search(query, inverted_index)
            print(result_articles)
            print(f"\nFound {len(result_articles)} results\n")
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
