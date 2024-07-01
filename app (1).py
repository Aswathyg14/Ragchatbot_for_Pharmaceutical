import streamlit as st
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb

# Load model and tokenizer
base_model = "microsoft/Phi-3-mini-128k-instruct"
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0},
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model)

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    pages = [page.extract_text() for page in reader.pages]
    return '\n'.join(pages)

# Function to get overlapping chunks
def get_overlapped_chunks(text, chunksize, overlapsize):
    return [text[i:i+chunksize] for i in range(0, len(text), chunksize-overlapsize)]

# Function to retrieve from vector DB
def retrieve_vector_db(query, collection, embedding_model, n_results=3):
    results = collection.query(
        query_embeddings=embedding_model.encode(query).tolist(),
        n_results=n_results
    )
    return results['documents']

# Function to get response from model
def get_phi3_chat_response(question, context, max_new_tokens=1000, end_token='</s>'):
    prompt = [
        {
            "role": "system",
            "content": "You are a friendly chatbot and answer the user's question based on the given context. If the answer is not in the context provided, just answer `I don't know`. Please do not give any additional information. Do not use context distillation."
        },
        {
            "role": "user",
            "content": f"Context: {context}\nQuestion: {question}"
        }
    ]
    tokenized_chat = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(tokenized_chat, max_new_tokens=max_new_tokens, temperature=0.00001)
    response = tokenizer.decode(outputs[0])

    start_idx = response.find("Question:") + len(f"Question: {question}")
    end_idx = response.find(end_token, start_idx)

    if end_idx != -1:
        extracted_response = response[start_idx:end_idx].strip()
    else:
        extracted_response = response[start_idx:].strip()

    unwanted_characters = ['\n', '\n\n1', '\n2', '</s>', '']
    for char in unwanted_characters:
        extracted_response = extracted_response.replace(char, '')

    if "assistant" in extracted_response:
        extracted_response = extracted_response.split("assistant")[-1].strip()

    return extracted_response

# Streamlit app
st.title("RAG with Phi-3 Model")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        document = extract_text_from_pdf(uploaded_file)

    chunks = get_overlapped_chunks(document, 1000, 100)
    chunk_embeddings = embedding_model.encode(chunks)
    
    chroma_client = chromadb.Client()
    collection_name = "rag_phi3"
    # Check if the collection exists
    if collection_name in [coll.name for coll in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(name=collection_name)
        collection.add(
            embeddings=chunk_embeddings,
            documents=chunks,
            ids=[str(i) for i in range(len(chunks))]
        )

    st.success("PDF text extracted and embedded!")

    # User query
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Retrieving context and generating response..."):
            retrieved_results = retrieve_vector_db(query, collection, embedding_model, n_results=5)
            context = '\n\n'.join(retrieved_results[0])
            response = get_phi3_chat_response(query, context, max_new_tokens=300)
            st.text_area("Response:", value=response, height=200)

# # Run the app
# if __name__ == "__main__":
#     st.run()