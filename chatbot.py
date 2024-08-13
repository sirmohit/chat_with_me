import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import faiss
import torch

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load a SentenceTransformer model for embedding generation
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Dummy knowledge base (documents)
documents = [
    "Flan-T5 is a variant of Google's T5 model that has been fine-tuned with instruction-based data.",
    "RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval of documents with text generation.",
    "Streamlit is an open-source app framework used to deploy machine learning models easily.",
]

# Generate embeddings for the knowledge base documents
doc_embeddings = embedder.encode(documents, convert_to_tensor=True)

# Create an FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.cpu().numpy())

# Streamlit app
st.title("RAG-based Chatbot with Flan-T5")

def generate_response(question):
    # Encode the user's question
    question_embedding = embedder.encode([question], convert_to_tensor=True)

    # Retrieve the most similar document
    top_k = 1
    distances, indices = index.search(question_embedding.cpu().numpy(), top_k)
    retrieved_doc = documents[indices[0][0]]

    # Prepare the input for Flan-T5
    input_text = f"question: {question} context: {retrieved_doc}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the response
    outputs = model.generate(input_ids)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Streamlit interface
user_input = st.text_input("Ask me anything:")
if user_input:
    response = generate_response(user_input)
    st.write("Bot:", response)
