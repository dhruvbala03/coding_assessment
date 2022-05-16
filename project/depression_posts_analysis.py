from audioop import minmax
import csv
from collections import defaultdict
from datetime import datetime
import json
import math
from posixpath import split
import re
from statistics import mean, variance
from collections import Counter
from black import STDIN_PLACEHOLDER
from textblob import TextBlob


# References:
## 1) https://www.geeksforgeeks.org/working-csv-files-python/
## 2) https://stackoverflow.com/questions/30598350/unicodedecodeerror-charmap-codec-cant-decode-byte-0x8d-in-position-7240-cha
## 3) https://www.adamsmith.haus/python/answers/how-to-count-the-number-of-unique-values-in-a-list-in-python
## 4) https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
## 5) https://www.programiz.com/python-programming/methods/built-in/filter
## 6) https://www.geeksforgeeks.org/find-k-frequent-words-data-set-python/
## 7) https://raw.githubusercontent.com/zelandiya/RAKE-tutorial/master/data/stoplists/SmartStoplist.txt
## 8) https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis 


# READ CSV FILE

filename = "resources/depression-sampled.csv"
fields = []
rows = []

authors = []
dates = []
posts = []

important_words = []

with open(filename, "r", encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    # extract field names from first row
    fields = next(csvreader)
    # extract data row-wise
    for row in csvreader:
        # skip rows with removed/deleted/empty or unusable post data
        if (
            (row[7] == "[removed]")
            or (row[7] == "[deleted]")
            or (row[7] == "")
            or (row[1] == "deleted")
            or (len(row) != 9)
        ):
            continue
        try:
            rows.append(row)
            authors.append(row[1])  # extract author
            dates.append(int(row[2]))  # extract timestamp
            posts.append(row[7])  # extract post text
        except:
            continue


# TOTAL NUMBER OF POSTS

num_posts = len(posts)
print(
    f"\nTotal number of posts (excluding deleted, removed, and empty posts): {num_posts}"
)


# TOTAL NUMBER OF UNIQUE AUTHORS

num_unique_authors = len(set(authors))
print(f"\nTotal number of unique authors: {num_unique_authors}")


# AVERAGE POST LENGTH

def extract_words(post):
    # replace punctuation with spaces
    post_edited = re.sub(pattern="[-\\n,.!]+", repl=" ", string=post)
    # account for weird apostrophe
    post_edited = re.sub(pattern="’+", repl="'", string=post_edited)
    # remove extra spaces
    post_edited = re.sub(pattern=" +", repl=" ", string=post_edited)
    return post_edited.split(" ")

word_counts = []

# unimportant words (for later use)
stoplist_dir = "resources/stoplist.txt"
stoplist_file = open(stoplist_dir, "r")
stop_words = stoplist_file.read().split("\n")

for post in posts:
    new_words = extract_words(post)
    word_counts.append(len(new_words))  # record word count
    # record important words (for later use)
    important_words.extend(
        [word for word in new_words if word.lower() not in stop_words]
    )

avg_post_length = round(mean(word_counts))
print(f"\nAverage post length (measured in word count): {avg_post_length}")


# DATE RANGE OF DATASET

earliest_date = datetime.fromtimestamp(min(dates))
latest_date = datetime.fromtimestamp(max(dates))
print(f"\nDate range of dataset\n\tEarliest: {earliest_date}\n\tLatest: {latest_date}")


# TOP 20 IMPORTANT WORDS

counter = Counter(important_words)
top_20_important_words = [x[0] for x in counter.most_common(20)]
print(f"\nTop 20 important words: {top_20_important_words}")


# TONE & SUBJECTIVITY ANALYSIS

sents = []
subjs = []

for post in posts:
    blob = TextBlob(post)
    sents.append(blob.sentiment.polarity)      # -1 to 1, more positive means happier sentiment
    subjs.append(blob.sentiment.subjectivity)  # 0 to 1, more positive means less factual

avg_sents = mean(sents)
sd_sents = math.sqrt(variance(sents))

avg_subj = mean(subjs)
sd_subj = math.sqrt(variance(subjs))

print(f"\nTone Analysis")
print(f"\tSentiment Polarity (-1 to 1, more positive means happier sentiment)\n\t\tmean: {avg_sents}\n\t\tstandard deviation: {sd_sents}")
print(f"\tSubjectivity (0 to 1, more positive means more opinionated sentiment)\n\t\tmean: {avg_subj}\n\t\tstandard deviation: {sd_subj}")
