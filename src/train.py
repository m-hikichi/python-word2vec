from gensim.models import word2vec


def main():
    sentences = word2vec.Text8Corpus("/tmp/jawiki-latest-pages-articles.text8")
    model = word2vec.Word2Vec(sentences, vector_size=200, window=15, min_count=20)
    model.save("../model/jawiki-latest-pages-articles.model")


if __name__=="__main__":
    main()
