import csv
import os

base_path = "/Users/sujaybhowmick/development/courses/mlnd/MLND-Capstone/twitters-sentiment-analysis"

sentiment_dict = {"neutral": 0, "positive": 1, "negative": -1}
write_rows = []


def write_csv_file(write_rows, file_path):
    with open(file_path, mode="a", encoding="latin-1") as appendabe_file:
        writer = csv.writer(appendabe_file)
        for row in write_rows:
            writer.writerow(row)


def process_file_row(csv_row, file):
    twitter_id = csv_row[0]
    content = csv_row[2]
    sentiment = str(csv_row[3]).lower().strip()
    if sentiment in sentiment_dict.keys():
        sentiment = sentiment_dict[sentiment]
        if len(write_rows) != 1000:
            write_rows.append([twitter_id, content, sentiment])
        else:
            write_csv_file(write_rows, base_path + "/data_formatted" + "/" + file)
            write_rows.clear()
    else:
        print(twitter_id, sentiment)


def process_raw_csv_files(dir_path="/data"):
    files = [filename for filename in os.listdir(dir_path) if filename.endswith(".csv")]
    file_out = "twitter_tweets.csv"
    for file in files:
        with open(dir_path + "/" + file, encoding='latin-1',) as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)
            for row in reader:
                if str(row[0]).find("_") == -1:
                    process_file_row(row, file_out)


if __name__ == "__main__":
    process_raw_csv_files(base_path + "/data")
