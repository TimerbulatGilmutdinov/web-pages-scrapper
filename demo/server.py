import os

from flask import Flask, render_template, request, Response, abort
from vector_search.vector_search import find_top_articles, preload_tf_idf_vectors

app = Flask("searching_server")
articles_tf_idf_vectors, idf = preload_tf_idf_vectors("../tf_idf/lemmas")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", str).strip()

    if not query:
        render_template("index.html")

    count, results, elapsed_time = find_top_articles(
        query=query,
        articles_vectors=articles_tf_idf_vectors,
        idf=idf,
        pages_dir="../scrapper/saved_pages",
        articles_count=10
    )

    return render_template(
        template_name_or_list="results.html",
        query=query,
        results=results,
        count=count,
        elapsed_time=elapsed_time,
    )


@app.route('/articles/<filename>')
def load_article(filename):
    article_path = os.path.join('../scrapper/saved_pages', filename)

    if not os.path.exists(article_path):
        abort(404)

    with open(article_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return Response(content, mimetype='text/html')

if __name__ == "__main__":
    app.run(debug=True, port=8080)
