from typing import Dict, List
import re
import pandas


def clean_data(tweets) -> List[str]:
    cleaned_tweets: List[str] = []

    for i, tweet in enumerate(tweets):
        clean = tweet.strip()
        clean = re.sub(r'', '', clean)
        # remove user mentions
        clean = re.sub(r'\B@[^\s]+\b', '', clean)

        # remove hashtags
        clean = re.sub(r'\B#[^\s]+\b', '', clean)

        # remove special chars, emojis, etc.
        clean = re.sub(r'[^a-zA-Z0-9.\s]', '', clean)

        clean = clean.lower()

        cleaned_tweets.append(clean)
        if i <= 5:
            print(f'Original: {tweet}')
            print(f'Cleaned:  {clean}')

    return cleaned_tweets



def parse_file(file_name: str):

    # Read the file in
	data: pandas.DataFrame = pandas.read_csv(file_name, sep=',', header=0)
	# print(data)

	# output_file: str = 'clean_trained.csv'
	data['tweet'] = clean_data(data['tweet'])

	# with open(file_name, 'r') as f:
	#	clean_line = re.sub('[^\\w\\d.\' ]', '', str.lower(line.strip()))


parse_file('train.csv')

