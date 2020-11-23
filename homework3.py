import gensim
import gzip
import matplotlib.pyplot as plt
import pandas
import re

from typing import List
from gensim.models import Word2Vec
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

stop_words = set(stopwords.words('english'))


def parse_file(file_name: str) -> DataFrame:
    # Read the file in
    data: DataFrame = pandas.read_csv(file_name, sep=',', header=0)
    data['tweet'] = clean_data(data['tweet'])
    return data


def clean_data(tweets) -> List[str]:
    cleaned_tweets: List[str] = []

    for i, tweet in enumerate(tweets):
        clean = tweet.strip()
        # remove user mentions
        clean = re.sub(r'\B@[^\s]+\b', '', clean)
        # remove hashtags
        clean = re.sub(r'\B#[^\s]+\b', '', clean)
        # remove '&amp'
        clean = re.sub(r'&amp', '', clean)
        # remove special chars, emojis, etc.
        clean = re.sub(r'[^\'a-zA-Z0-9.\s]', '', clean)

        clean = clean.strip().lower()

        clean = gensim.utils.simple_preprocess(clean)

        # Remove stopwords
        clean = [word for word in clean if word not in stop_words]

        clean = ' '.join(clean)
        cleaned_tweets.append(clean)

    return cleaned_tweets


def part0(file_name: str) -> DataFrame:
    data = parse_file(file_name)
    return data


def part1(data: DataFrame):
    model: Word2Vec = Word2Vec(data['tweet'], size=150, window=10, min_count=2, workers=10)
    model.train(data['tweet'], total_examples=len(data['tweet']), epochs=10)
    part1_q1_q3()
    part1_q2(data)


def part1_q2(data: DataFrame):
    non_hate = FreqDist()
    for i, tweet in enumerate(data['tweet']):
        if data['label'][i] == 0:
            for word in tweet:
                non_hate[word] += 1

    top_words = [word[0] for word in non_hate.most_common(10)]
    top_counts = [word[1] for word in non_hate.most_common(10)]
    print(non_hate.most_common(10))
    plt.bar(top_words, top_counts, color='green')
    plt.show()

    hate = FreqDist()
    for i, tweet in enumerate(data['tweet']):
        if data['label'][i] == 1:
            for word in tweet:
                hate[word] += 1

    top_words = [word[0] for word in hate.most_common(10)]
    top_counts = [word[1] for word in hate.most_common(10)]
    print(hate.most_common(10))
    plt.bar(top_words, top_counts, color='red')

    plt.show()


def part1_q1_q3():
    imdb = list(read_imdb('imdb.txt.gz'))
    print('Training...')
    model = gensim.models.Word2Vec(imdb, size=150, window=10, min_count=2, workers=10)
    model.train(imdb, total_examples=len(imdb), epochs=10)
    print('Done training')

    print(f"Most similar to 'hate': {model.wv.most_similar(positive=['hate'])[:1]}")
    print(f"Most similar to 'like': {model.wv.most_similar(positive=['like'])[:1]}")

    print(model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[:2])
    print(f"Analogy for love + life - hate: {model.wv.most_similar(positive=['love', 'life'], negative=['hate'])[:1]}")


def read_imdb(input_file: str) -> List[str]:
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print("read {0} reviews".format(i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess(line)


def part2(train: DataFrame):
    print('Part 2:')
    part2_q1(train)
    part2_q2(train)
    part2_q3(train)
    return


def part2_q1(data: DataFrame):
    model = MultinomialNB(alpha=1)
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True)
    apply_model(model, vectorizer, data)


def part2_q2(data: DataFrame):
    model = LogisticRegression()
    vectorizer = CountVectorizer(binary=True)
    apply_model(model, vectorizer, data)


def part2_q3(data: DataFrame):
    model = LogisticRegression(multi_class='ovr')
    vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True)
    apply_model(model, vectorizer, data)


def apply_model(model, vectorizer: CountVectorizer, data: DataFrame):
    train_data = vectorizer.fit_transform(data['tweet']).toarray()
    train_label = data['label'].to_numpy()

    data_train, data_test, label_train, label_test = train_test_split(train_data, train_label, test_size=0.238310853)
    label_test = label_test.ravel()
    model.fit(data_train, label_train)

    pred_labels = model.predict(data_test)
    print(confusion_matrix(label_test, pred_labels))
    print(classification_report(label_test, pred_labels))
    print(accuracy_score(label_test, pred_labels))
    f1 = f1_score(label_test, pred_labels)
    print(f'f1-score: {f1}')


def main():
    data: DataFrame = part0('train.csv')
    part1(data)
    part2(data)


main()
