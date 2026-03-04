from dotenv import load_dotenv

import csv
import os
import subprocess
import sys
import requests
import json
import tiktoken

from bs4 import BeautifulSoup
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from bertopic.representation import OpenAI
from openai import OpenAI as OpenAIClient
from jinja2 import Environment, FileSystemLoader
from io import BytesIO
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

import transformers

load_dotenv()

transformers.logging.set_verbosity_error()

if len(sys.argv) == 1:
  sys.stderr.write(f"Usage: {sys.argv[0]} <taxon-base-path>\n")
  sys.exit(1)

taxon_base_path = sys.argv[1]

print(f"Generating query for taxon with path: {taxon_base_path}")

env = Environment(loader=FileSystemLoader("."))
template = env.get_template("query.sql.jinja")
sql = template.render(taxon_base_path=taxon_base_path)
with open("query.sql", "w") as text_file:
    text_file.write(sql)
    text_file.write("\n")

print("Querying for content items...")

input_dir_path = f"input{taxon_base_path}"
input_file_path = f"{input_dir_path}/input.csv"
os.makedirs(input_dir_path, exist_ok=True)

output_dir_path = f"output{taxon_base_path}"
output_file_path = f"{output_dir_path}/output.csv"
os.makedirs(output_dir_path, exist_ok=True)

subprocess.run("govuk-docker up -d content-store-lite", shell=True, check=True)
subprocess.run(f"docker exec -i govuk-docker-content-store-lite-1 rails db < query.sql > {input_file_path}", shell=True, check=True)
subprocess.run("govuk-docker down content-store-lite", shell=True, check=True)

print(f"Writing content items to {input_file_path}...")

content_items = []

print("Loading content items...")

def extract_text_from_pdf_attachment(url):
    print(f"Fetching PDF attachment: {url}")
    response = requests.get(url)
    pdf_bytes = BytesIO(response.content)
    try:
        pdf_reader = PdfReader(pdf_bytes)
    except PdfReadError as e:
        print(f"Error reading PDF: {e}")
        return ""

    full_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        full_text += text + "\n\n"
    return full_text

def pdf_attachment_urls(attachments_json):
    try:
        attachments_data = json.loads(attachments_json)
    except json.JSONDecodeError:
        return []

    pdf_urls = [
        attachment["url"]
        for attachment in attachments_data
        if attachment.get("attachment_type") == "file" and attachment.get("content_type") == "application/pdf"
    ]
    return pdf_urls

def extract_text_from_html_attachment(path):
    print(f"Fetching HTML attachment: {path}")
    response = requests.get(f"https://www.gov.uk/api/content{path}")
    attachment = response.json()
    title = attachment["title"]
    body = attachment["details"]["body"]
    body_text = BeautifulSoup(body, "html.parser").get_text(
        separator=" ", strip=True
    )
    combined_body_text = "\n".join([
        f"{title} {title}",
        body_text
    ]).strip()
    return combined_body_text

def html_attachment_paths(attachments_json):
    try:
        attachments_data = json.loads(attachments_json)
    except json.JSONDecodeError:
        return []

    paths = [
        attachment["url"]
        for attachment in attachments_data
        if attachment.get("attachment_type") == "html"
    ]
    return paths

with open(input_file_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        title = row["title"]
        body_text = BeautifulSoup(row["body"], "html.parser").get_text(
            separator=" ", strip=True
        )
        combined_body_text = "\n".join([
            f"{title} {title}",
            body_text,
            *[extract_text_from_pdf_attachment(url) for url in pdf_attachment_urls(row["attachments"])],
            *[extract_text_from_html_attachment(url) for url in html_attachment_paths(row["attachments"])]
        ]).strip()

        content_items.append({
            "id": row["id"],
            "base_path": row["base_path"],
            "title": row["title"],
            "text": combined_body_text,
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

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

representation_model = OpenAI(
  client,
  model="openai/gpt-4o-mini",
  chat=True,
  nr_docs=4,
  doc_length=2000,
  tokenizer=tokenizer
)

topic_model = BERTopic(
  vectorizer_model=vectorizer,
  representation_model=representation_model,
  calculate_probabilities=True
)

docs = [content_item["text"] for content_item in content_items]

topics, probs = topic_model.fit_transform(docs)

topic_model.visualize_topics().write_html(f"{output_dir_path}/topics.html")
topic_model.visualize_documents(docs).write_html(f"{output_dir_path}/documents.html")
topic_model.visualize_hierarchy().write_html(f"{output_dir_path}/hierarchy.html")

topic_content_items = defaultdict(list)
for content_item, topic_id, prob_array in zip(content_items, topics, probs):
    content_item_prob = prob_array[topic_id] if topic_id != -1 else 0
    topic_content_items[topic_id].append((content_item["id"], content_item["base_path"], content_item["title"], content_item_prob))

topic_info = topic_model.get_topic_info()
topic_names = topic_info.set_index("Topic")["Name"].to_dict()

with open(output_file_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["topic_number", "topic_name", "link", "probability", "content_item_id"])
    writer.writeheader()

    for topic_id in sorted(topic_content_items):
        if topic_id == -1:
            name = "Outlier"
        else:
            name = topic_names.get(topic_id, "Unknown").split("_", 1)[-1]

        for content_item_id, base_path, title, prob in topic_content_items[topic_id]:
            writer.writerow({
              "topic_number": topic_id,
              "topic_name": name,
              "link": f'=HYPERLINK("https://gov.uk{base_path}","{title.replace('"', '""')}")',
              "probability": f"{prob:.3f}",
              "content_item_id": content_item_id
            })

print(f"Written to {output_file_path}")
