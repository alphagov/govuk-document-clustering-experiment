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

content_items = []

print("Loading content items...")

with open("input.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = row["title"]
        body_text = BeautifulSoup(row["body"], "html.parser").get_text(
            separator=" ", strip=True
        )
        body_text_combined_with_double_weighted_title = f"{title} {title} {body_text}".strip()
        content_items.append({
            "id": row["id"],
            "text": body_text_combined_with_double_weighted_title,
        })

print(f"Loaded {len(content_items)} content items")

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
  calculate_probabilities=True
)
topics, probs = topic_model.fit_transform([content_item["text"] for content_item in content_items])
topic_content_items = defaultdict(list)
for content_item, topic_id, prob_array in zip(content_items, topics, probs):
    content_item_prob = prob_array[topic_id] if topic_id != -1 else 0
    topic_content_items[topic_id].append((content_item["id"], content_item_prob))

for topic_id in sorted(topic_content_items):
    if topic_id == -1:
        keywords = "outlier"
    else:
        label = topic_model.get_topic(topic_id)
        keywords = ", ".join(keyword for keyword, _ in label)

    print(f"\nTopic {topic_id}  [{keywords}]:")

    for content_item_id, prob in topic_content_items[topic_id]:
        print(f"  {content_item_id}  (prob={prob:.3f})")
