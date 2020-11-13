from typing import List
import gensim
import gzip
import pandas
import re


def clean_data(tweets) -> List[str]:
    cleaned_tweets: List[str] = []

    for i, tweet in enumerate(tweets):
        clean = tweet.strip()

        # remove part indication of tweets (e.g. [1/2])
        clean = re.sub(r'\[\d/\d]', '', clean)

        # remove user mentions
        clean = re.sub(r'\B@[^\s]+\b', '', clean)

        # remove hashtags
        clean = re.sub(r'\B#[^\s]+\b', '', clean)

        # remove special chars, emojis, etc.
        clean = re.sub(r'[^a-zA-Z0-9.\s]', '', clean)

        # clean = clean.strip().lower()
        clean = gensim.utils.simple_preprocess(clean)

        cleaned_tweets.append(clean)
        # if i <= 5:
        #     print(f'Original: {tweet}')
        #     print(f'Cleaned:  {clean}')

    return cleaned_tweets


def parse_file(file_name: str):
    # Read the file in
    data: pandas.DataFrame = pandas.read_csv(file_name, sep=',', header=0)
    data['tweet'] = clean_data(data['tweet'])

    tweets: List[str] = list(data['tweet'])
    model = gensim.models.Word2Vec(tweets, size=150, window=10, min_count=2, workers=10)
    model.train(tweets, total_examples=len(tweets), epochs=10)
    print(f"Most similar to 'hate': {model.wv.most_similar(positive=['hate'])[:1]}")
    print(f"Most similar to 'like': {model.wv.most_similar(positive=['like'])[:1]}")
    # print(data)


parse_file('train.csv')

