import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import pymorphy2

eng_token_regex = re.compile(r'^[a-zA-Z]{2,}$')
rus_token_regex = re.compile(r'^[а-яА-ЯёЁ]{2,}$')

rus_lemmatizer = pymorphy2.MorphAnalyzer()
eng_lemmatizer = WordNetLemmatizer()


def lemmatize(tokens):
    lemmas = defaultdict(set)

    for token in tokens:
        if eng_token_regex.match(token):
            lemma = eng_lemmatizer.lemmatize(token)
        elif rus_token_regex.match(token):
            lemma = rus_lemmatizer.parse(token)[0].normal_form
        else:
            continue

        lemmas[lemma].add(token)

    return lemmas


def extract_tokens(text):
    eng_stop_words = set(stopwords.words('english'))
    rus_stop_words = set(stopwords.words('russian'))

    tokenized_text = word_tokenize(text)
    result_tokens = set()

    for token in tokenized_text:
        if eng_token_regex.fullmatch(token) and token not in eng_stop_words:
            result_tokens.add(token)
        elif rus_token_regex.fullmatch(token) and token not in rus_stop_words:
            result_tokens.add(token)

    return result_tokens


def lemmatize_page(file_path):
    filename = os.path.basename(file_path)
    raw_html = open(file_path, 'r', encoding='utf-8').read()

    soup = BeautifulSoup(raw_html, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)

    tokens = extract_tokens(text.lower())
    lemmas = lemmatize(tokens)

    tokens_file = os.path.join('tokens', filename)
    lemmas_file = os.path.join('lemmas', filename)

    with open(tokens_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(sorted(tokens)))

    with open(lemmas_file, 'w', encoding='utf-8') as file:
        for lemma, token_list in sorted(lemmas.items()):
            file.write(f"{lemma} {' '.join(sorted(token_list))}\n")

    print(f"[✓] {filename}: {len(tokens)} токенов, {len(lemmas)} лемм")


def start_lemmatizing():
    saved_pages_path = '../scrapper/saved_pages'

    for filename in os.listdir(saved_pages_path):
        try:
            lemmatize_page(os.path.join(saved_pages_path, filename))
        except Exception as e:
            print(f"[х] {filename}: {str(e)}")


if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    os.makedirs('tokens', exist_ok=True)
    os.makedirs('lemmas', exist_ok=True)

    start_lemmatizing()
