import csv
import re

base_dir = "/Users/sujaybhowmick/development/courses/mlnd/MLND-Capstone/twitter-sentiment-analysis"
preprocessed_tweets = []

def clean_tweet(tweet):
    #tweet = row[1]
    new_tweet = ''
    for word in tweet.split():
        # String preprocessing
        if re.match('^.*@.*', word):
            word = '<NAME/>'
        if re.match('^.*http[s]?://.*', word):
            word = '<LINK/>'
        word = word.replace('#', '<HASHTAG/> ')
        word = word.replace('&quot;', ' \" ')
        word = word.replace('&amp;', ' & ')
        word = word.replace('&gt;', ' > ')
        word = word.replace('&lt;', ' < ')
        new_tweet = ' '.join([new_tweet, word])
    tweet = new_tweet.strip().strip(".")
    return tweet


def clean_str(cleaned_tweet):
    """
    Tokenizes common abbreviations and punctuation, removes unwanted characters.
    Returns the clean string.
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", cleaned_tweet)
    string = re.sub(r'(.)\1+', r'\1\1', cleaned_tweet)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"“”¨«»®´·º½¾¿¡§£₤‘’", "", string)
    return string.strip().lower()


def preprocess_csv_file(file_in):
    with open(file_in, "r", encoding="latin-1") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            cleaned_tweet = clean_tweet(clean_str(row[1]))
            preprocessed_tweets.append([row[0], cleaned_tweet, row[2]])


if __name__ == "__main__":
    twitter_date_file_in = "/Users/sujaybhowmick/development/courses/mlnd/MLND-Capstone/twitter-sentiment-analysis/data_formatted/twitter_tweets.csv"
    preprocess_csv_file(twitter_date_file_in)
    with open(base_dir + "/preprocessed_tweets.csv", mode="a", encoding="latin-1") as csv_file_w:
        writer = csv.writer(csv_file_w)
        writer.writerow(["msg_id", "content", "label"])
        for row_w in preprocessed_tweets:
            writer.writerow(row_w)
    print("Total Preprocessed Tweets:", len(preprocessed_tweets))
    preprocessed_tweets.clear()
