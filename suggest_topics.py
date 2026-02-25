from dotenv import load_dotenv

import csv
from bs4 import BeautifulSoup
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

import transformers

load_dotenv()

transformers.logging.set_verbosity_error()

doc_ids, texts = [], []

print("Loading documents...")

with open("input.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = row["title"]
        body_text = BeautifulSoup(row["body"], "html.parser").get_text(
            separator=" ", strip=True
        )
        combined_with_double_weighted_title = f"{title} {title} {body_text}".strip()
        doc_ids.append(row["id"])
        texts.append(combined_with_double_weighted_title)

print(f"Loaded {len(texts)} documents")

vectorizer = CountVectorizer(
  stop_words="english",
  min_df=3,
  max_df=0.9,
  ngram_range=(1,2)
)

topic_model = BERTopic(
  vectorizer_model=vectorizer,
)
topics, probs = topic_model.fit_transform(texts)

for doc_id, topic_id in zip(doc_ids, topics):
    if topic_id == -1:
        keywords = "outlier"
    else:
        label = topic_model.get_topic(topic_id)
        keywords = ", ".join(keyword for keyword, _ in label)

    print(f"  {doc_id}  →  topic {topic_id:3d}  [{keywords}]")
