from dotenv import load_dotenv

import csv
import os
from bs4 import BeautifulSoup
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from bertopic.representation import OpenAI
from openai import OpenAI as OpenAIClient

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

client = OpenAIClient(
  api_key=os.environ.get("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
)

representation_model = OpenAI(
  client,
  model="openai/gpt-4o-mini",
  chat=True,
)

topic_model = BERTopic(
  vectorizer_model=vectorizer,
  representation_model=representation_model,
)
topics, probs = topic_model.fit_transform(texts)

topic_docs = defaultdict(list)
for doc_id, topic_id in zip(doc_ids, topics):
    topic_docs[topic_id].append(doc_id)

for topic_id in sorted(topic_docs):
  if topic_id == -1:
      keywords = "outlier"
  else:
      label = topic_model.get_topic(topic_id)
      keywords = ", ".join(keyword for keyword, _ in label)

  print(f"\nTopic {topic_id}  [{keywords}]:")

  for doc_id in topic_docs[topic_id]:
      print(f"  {doc_id}")
