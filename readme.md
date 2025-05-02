### Основы информационного поиска - 4с2s
Гильмутдинов Тимербулат 11-101

Clone repo:
```bash
git clone https://github.com/TimerbulatGilmutdinov/web-pages-scrapper.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Task 1 - web scrapper
Web scrapper used to scrap particular Habr hub articles 

#### How to run

```bash
cd /web-pages-scrapper
python scrapper.py
```

### Task 2 - lemmatizing and tokenizing
Lemmatizing and tokenizing saved Habr pages

#### How to run

```bash
cd /web-pages-scrapper/lemmatizer
python lemmatizer.py
```

### Task 3 - inverted index building
Building inverted index and searching lemmas with it

#### How to run

```bash
cd /web-pages-scrapper/inverted_index
python indexer.py
python searcher.py
```

### Task 4 - TF-IDF
Building TF-IDF

#### How to run

```bash
cd /web-pages-scrapper/tf_idf
python tf_idf.py
```

### Task 5 - Cos similarity search
Searching via cos vector similarity

#### How to run

```bash
cd /web-pages-scrapper/vector_search
python vector_search.py
```
