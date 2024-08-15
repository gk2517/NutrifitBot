import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cpu")


def generate_embedding(text):
    response = model.encode([text])
    return np.array(response[0])

import faiss
import numpy as np


class VectorStore:
  def __init__(self):
    self.documents = []
    self.embeddings = np.empty((0,384))  # Initialize as empty array

  def add_to_store(self, document):
    # Append the document to the list of documents
    self.documents.append(document)

    # Generate the embedding for the document
    response = generate_embedding(document.content)

    # Concatenate the response with the existing embeddings vertically
    self.embeddings = np.vstack((self.embeddings, response))

"""
Import docs generated in part 1.
Make sure you add docs.pkl to the files!
"""

import pickle

class Document:
  def __init__(self, title, url, content):
    self.title = title
    self.url = url
    self.content = content

with open('docs.pkl', 'rb') as file:
    docs = pickle.load(file)

store = VectorStore()

##for i in range(len(docs)):
  ##print(f"Processing {i}...")
  ##store.add_to_store(docs[i])

"""
create_index
"""

# Create an index with the same dimension as the embeddings
index = faiss.IndexFlatL2(store.embeddings.shape[1])

# Add the embeddings to the index
index.add(store.embeddings)

faiss.write_index(index, 'index.faiss')

import json
print(store.embeddings)
with open('json/output_data.json', 'w') as f:
    json.dump(store.embeddings.tolist(), f)