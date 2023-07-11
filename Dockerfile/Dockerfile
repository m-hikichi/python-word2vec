FROM python:3.10

# download
RUN wget -P /tmp https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
# unzip
WORKDIR /tmp
RUN apt-get update && apt install bzip2
RUN bzip2 -d /tmp/jawiki-latest-pages-articles.xml.bz2
# extract sentences for each article
RUN pip install wikiextractor
RUN python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml
RUN cat text/*/* > jawiki-latest-pages-articles.txt
# delete html-tag
RUN sed -i '/^<[^>]*>$/d' jawiki-latest-pages-articles.txt
# wakati
RUN apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
RUN mecab -Owakati jawiki-latest-pages-articles.txt -o jawiki-latest-pages-articles.text8 -b 200000

# package install for natural language processing
RUN pip install gensim
