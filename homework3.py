from typing import List
import numpy.random as rng
import gensim
import gzip
from gensim.models import Word2Vec
from nltk.probability import FreqDist
from nltk.corpus import brown
from nltk.corpus import stopwords
import nltk
import numpy as np
from pandas import DataFrame
# import nltk
import matplotlib.pyplot as plt
import pandas
import re
import sklearn
from sklearn.naive_bayes import MultinomialNB

stop_words = set(stopwords.words('english'))


def parse_file(file_name: str) -> DataFrame:
    # Read the file in
    data: DataFrame = pandas.read_csv(file_name, sep=',', header=0)
    data['tweet'] = clean_data(data['tweet'])
    return data


def clean_data(tweets) -> List[List[str]]:
    cleaned_tweets: List[List[str]] = []

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

        cleaned_tweets.append(clean)

    return cleaned_tweets


def part0(file_name: str) -> DataFrame:
    data = parse_file(file_name)
    return data


def part1(data: DataFrame):
    # print(data['tweet'])
    model: Word2Vec = Word2Vec(data['tweet'], size=150, window=10, min_count=2, workers=10)
    model.train(data['tweet'], total_examples=len(data['tweet']), epochs=10)
    part1_q1(model)
    part1_q2(data)
    part1_q3()


def part1_q1(model: Word2Vec):
    print(f"Most similar to 'hate': {model.wv.most_similar(positive=['hate'])[:1]}")
    print(f"Most similar to 'like': {model.wv.most_similar(positive=['like'])[:1]}")


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


def part1_q3():
    imdb = list(read_imdb('imdb.txt.gz'))
    print('Training...')
    model = gensim.models.Word2Vec(imdb, size=150, window=10, min_count=2, workers=10)
    model.train(imdb, total_examples=len(imdb), epochs=10)
    print('Done training')
    print(model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[:2])
    print(f"Analogy for love + life - hate: {model.wv.most_similar(positive=['love', 'life'], negative=['hate'])[:1]}")


def read_imdb(input_file: str) -> List[str]:
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print("read {0} reviews".format(i))
                # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess(line)


def test_accuracy(model, test: DataFrame):
    print('Testing Accuracy for Naive Bayes model:')
    correct = 0
    for i, tweet in enumerate(test['tweet']):
        predicted = model.predict(tweet)
        if predicted == test['label'][i]:
            correct += 1

    accuracy: float = correct / len(test['tweet'])
    print(f'Accuracy: {accuracy}')
    return


def part2(train: DataFrame):
    print('Part 2:')
    part2_q1(train)
    test_tweets = parse_file('test.csv')
    # Naive Bayes with add-1 smoothing using binary bag-of-ngrams features
    clf = MultinomialNB()
    x = np.vstack(train['tweet'])
    # y = train['label'].astype(np.float).to_numpy()
    y = np.array(train['label']).astype(np.int)
    print(y)
    print(type(y[0]))

    clf.fit(x, y)
    # clf.fit(train['tweet'], train['label'])
    # test_accuracy(clf, test_tweets)
    return


def part2_q1(model: DataFrame):
    return


def main():
    data: DataFrame = part0('train.csv')
    part1(data)
    # part2(data)


main()
