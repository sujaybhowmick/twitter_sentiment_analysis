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
from concurrent import futures
import grpc

import sentiments_pb2 as sentiment_types
import sentiments_pb2_grpc as sentiments_service_grpc


class SentimentService(sentiments_service_grpc.SentimentsServicer):
    SENTIMENT_LABELS = ['negative', 'positive']

    VOCAB_SIZE = 30000
    MAX_LEN = 22

    def __init__(self, config):
        self.logger = logging.getLogger(self.__class__.__name__)
        base_path = config['app']['data_dir']
        models_path = config['app']['models_dir']

        self.logger.info("base_path: %s", base_path)
        self.logger.info("models_path: %s", models_path)

        df = pd.read_csv(base_path + "/preprocessed_tweets.csv", encoding='latin-1')
        df = df.drop(columns=['msg_id'])
        train, test = train_test_split(df, test_size=0.2)

        with open(models_path + "/tokenizer.pickle", "rb") as pickled_tokenizer:
            self.tokenizer = pickle.load(pickled_tokenizer)

        self.tokenizer.fit_on_texts(train['content'].values)
        x_train_seq = self.tokenizer.texts_to_sequences(train['content'].values)
        x_test_seq = self.tokenizer.texts_to_sequences(test['content'].values)

        x_train = sequence.pad_sequences(x_train_seq, maxlen=SentimentService.MAX_LEN, padding="post", value=0)
        x_test = sequence.pad_sequences(x_test_seq, maxlen=SentimentService.MAX_LEN, padding="post", value=0)

        y_train, y_test = train['label'].values, test['label'].values

        with open(models_path + '/model.json', 'r') as f:
            json = f.read()
        self.model = model_from_json(json)

        # load model weights
        self.model.load_weights(models_path + "/weights-improvement-01-0.79.hdf5")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=32, epochs=2, validation_split=0.1, verbose=0)
        # Evaluate the model
        score, acc = self.model.evaluate(x_test, y_test, batch_size=32)
        self.logger.info("%s: %.2f%%" % (self.model.metrics_names[1], acc * 100))

    def predict(self, tweet):
        tweet_words_array = self.tokenizer.texts_to_sequences([tweet])
        tweet_words_array = sequence.pad_sequences(tweet_words_array, maxlen=SentimentService.MAX_LEN, padding="post",
                                                   value=0)
        score = self.model.predict(tweet_words_array)[0][0]
        prediction = SentimentService.SENTIMENT_LABELS[self.model.predict_classes(tweet_words_array)[0][0]]
        return tweet, prediction, score


ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SentimentScoreServer(object):
    def __init__(self, sentiment_service, port, max_workers):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sentiment_score_service = sentiment_service
        self.port = port
        self.instance = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    def start(self):
        sentiments_service_grpc.add_SentimentsServicer_to_server(self.sentiment_score_service, self.instance)
        self.instance.add_insecure_port('[::]:%d' % self.port)
        self.instance.start()
        self.logger.info('Server is ready at port %d', self.port)

    def stop(self):
        self.instance.stop(0)
        self.logger.info('Server was stopped')


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

    sentiment_score_server = SentimentScoreServer(sentiment_service, port=config['grpc']['port'],
                                                  max_workers=config['grpc']['max_workers'])
    sentiment_score_server.start()
    keep_server_alive(sentiment_score_server)


if __name__ == "__main__":
    main(sys.argv)
