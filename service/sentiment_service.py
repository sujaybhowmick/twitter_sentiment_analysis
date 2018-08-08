from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

import pandas as pd
import pickle
import logging
import logging.config
import os
import yaml
import time
import sys

import sentiments_pb2 as sentiment_types
import sentiments_pb2_grpc as sentiments_service_grpc


class SentimentService(sentiments_service_grpc.SentimentsServicer):
    SENTIMENT_LABELS = ['negative', 'positive']

    VOCAB_SIZE = 30000
    MAX_LEN = 22

    def __init__(self, config):
        base_path = config['app']['base_dir']
        df = pd.read_csv(base_path + "data/preprocessed_tweets.csv", encoding='latin-1')
        df = df.drop(columns=['msg_id'])
        train, test = train_test_split(df, test_size=0.2)

        with open(base_path + "/models/tokenizer.pickle", "rb") as pickled_tokenizer:
            self.tokenizer = pickle.load(pickled_tokenizer)

        self.tokenizer.fit_on_texts(train['content'].values)
        x_train_seq = self.tokenizer.texts_to_sequences(train['content'].values)
        x_test_seq = self.tokenizer.texts_to_sequences(test['content'].values)

        x_train = sequence.pad_sequences(x_train_seq, maxlen=SentimentService.MAX_LEN, padding="post", value=0)
        x_test = sequence.pad_sequences(x_test_seq, maxlen=SentimentService.MAX_LEN, padding="post", value=0)

        y_train, y_test = train['label'].values, test['label'].values

        with open(base_path + '/models/model.json', 'r') as f:
            json = f.read()
        self.model = model_from_json(json)

        # load model weights
        self.model.load_weights(base_path + "/models/cnn_twitter_sentiment_weights.h5")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=32, epochs=2, validation_split=0.1, verbose=0)
        # Evaluate the model
        score, acc = self.model.evaluate(x_test, y_test, batch_size=32)
        print("%s: %.2f%%" % (self.model.metrics_names[1], acc * 100))

    def predict(self, tweet):
        tweet_words_array = self.tokenizer.texts_to_sequences([tweet])
        tweet_words_array = sequence.pad_sequences(tweet_words_array, maxlen=SentimentService.MAX_LEN, padding="post",
                                                   value=0)
        score = self.model.predict(tweet_words_array)[0][0]
        prediction = SentimentService.SENTIMENT_LABELS[self.model.predict_classes(tweet_words_array)[0][0]]
        return tweet, prediction, score


ONE_DAY_IN_SECONDS = 60 * 60 * 24


def keep_server_alive(sentiment_score_server):
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        sentiment_score_server.stop()


def load_yaml_file(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


def get_config_file_path(file_name):
    return os.path.join(os.path.realpath(os.path.dirname(__file__)), 'conf', file_name)


LOG_CONFIG_FILE = 'logging.yml'
APP_CONFIG_FILE = 'application.yml'


def main(argv):
    logging.config.dictConfig(load_yaml_file(get_config_file_path(LOG_CONFIG_FILE)))
    config = load_yaml_file(get_config_file_path(APP_CONFIG_FILE))
    sentiment_service = SentimentService(config)

if __name__ == "__main__":
    main(sys.argv)