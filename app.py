from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
app = Flask(__name__, template_folder='templates')
@app.route('/')
def index():
    return render_template('index.html')



    # Load the JSON data

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device="cpu")  



class Document:
    def __init__(self, title, url, content):
        self.title = title
        self.url = url
        self.content = content

def generate_embedding(text):
    response = model.encode([text])
    return np.array(response[0])




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



        
def load_object(filename):
    with open(filename ,'rb') as f:
        loaded = pickle.load(f)
        return loaded 

def load_json(fname):
    with open(fname ,'r') as f:
        loadedjson = json.load(f)
        return loadedjson  



@app.route('/search', methods=['POST', 'GET'])

    
def search():
    if request.method == "POST":
  
     query = request.form['query']
     
    myresults = []
    docs = load_object('docs.pkl')
        
    store = []
    store = VectorStore()
    myvector = load_json('output_data.json')

    feature_batch=np.array(myvector)

# Create an index with the same dimension as the embeddings
    index = faiss.IndexFlatL2(feature_batch.shape[1])

# Add the embeddings to the index


    index.add(feature_batch)

    faiss.write_index(index, 'index.faiss')



# Generate embedding for the given query




# Set the similarity threshold
    similarityThreshold = 1

    # Generate embedding for the given query
    query_embedding = generate_embedding(query)

# Search for similar embeddings in the index
    distances, results = index.search(np.array([query_embedding]), k=4)

    # Filter the results based on the similarity threshold
    filtered_results = []
    for i, distance in zip(results[0], distances[0]):
        if distance <= similarityThreshold:
            filtered_results.append(i)

    # Print the content of the documents
    
    for i in filtered_results:
        
        myresults.append(docs[i].content.replace("#", '').replace("Picture of beef:", '').replace(" iStock/Getty Images", '')) 
       
       
        
    
    
    if query :
     return jsonify({'output': myresults })
    return jsonify({'error' : 'Error!'})
# Perform a simple search by checking if the query is in any data item
   
    return render_template('index.html', myresults=myresults)

if __name__ == '__main__':
    
    app.run()
