import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from tqdm import tqdm

# Initialize Elasticsearch and LLM clients
es_client = Elasticsearch('http://localhost:9200')
client = Groq(api_key="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")

# Function to create a keyword index
def create_keyword_index(es_client, index_name):
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
            }
        },
    }
    es_client.indices.create(index=index_name, body=index_settings)
    st.success(f"Keyword index '{index_name}' created successfully.")

# Function to create a vector index
def create_vector_index(es_client, index_name):
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "id": {"type": "keyword"},
                "question_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
                "text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
                "question_text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }
    es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, body=index_settings)
    st.success(f"Vector index '{index_name}' created successfully.")

# Function to encode and index documents for vector search
def encode_and_index_documents(es_client, index_name, model, documents):
    for doc in tqdm(documents, desc="Encoding documents"):
        question = doc['question']
        text = doc['text']
        qt = question + ' ' + text
        doc['question_vector'] = model.encode(question).tolist()
        doc['text_vector'] = model.encode(text).tolist()
        doc['question_text_vector'] = model.encode(qt).tolist()

    for doc in tqdm(documents, desc="Indexing documents"):
        try:
            es_client.index(index=index_name, document=doc)
        except Exception as e:
            st.error(f"Error indexing document {doc['id']}: {e}")

# Function for keyword search
def keyword_elastic_search(es_client, index_name, query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {
                    "term": {"course": "data-engineering-zoomcamp"}
                },
            }
        },
    }
    response = es_client.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]

# Function for vector search
def vector_elastic_search(es_client, index_name, field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {"term": {"course": course}},
    }
    search_query = {"knn": knn, "_source": ["text", "section", "question", "course", "id"]}
    es_results = es_client.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in es_results["hits"]["hits"]]

# Function to build the prompt for LLM
def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()
    context = "\n\n".join([f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}" for doc in search_results])
    return prompt_template.format(question=query, context=context).strip()

# Function to generate the answer using LLM
def llm(prompt, client, llm_model):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=llm_model,
    )
    return response.choices[0].message.content

# Function to handle the RAG process
def rag(query, index_name="course-questions", use_vector_search=False, vector_field=None, vector=None, llm_model="llama3-8b-8192"):
    if use_vector_search and vector_field and vector is not None:
        search_results = vector_elastic_search(es_client, index_name, vector_field, vector, "data-engineering-zoomcamp")
    else:
        search_results = keyword_elastic_search(es_client, index_name, query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, client, llm_model)
    return answer

# Streamlit UI
st.title("RAG System with Elasticsearch and LLM")

# Select the type of search
search_type = st.selectbox("Select Search Type", ["Keyword Search", "Vector Search"])

# Select the LLM model
llm_model = st.selectbox("Select LLM Model", ["llama3-8b-8192", "llama2-7b", "other-models"])

# Input for user query
query = st.text_input("Enter your query")

# Optionally select vector field (for vector search)
# vector_field = None
# if search_type == "Vector Search":
#     #vector_field = st.text_input("Enter vector field", value="question_text_vector")

# Submit button
if st.button("Get Answer"):
    # Ensure the index exists
    index_name = "course-questions"
    if not es_client.indices.exists(index=index_name):
        create_vector_index(es_client, index_name)

    # Perform RAG process based on selected search type
    if search_type == "Vector Search":
        # Use a pre-encoded vector for the query
        model = SentenceTransformer("all-mpnet-base-v2")
        query_vector = model.encode(query).tolist()
        answer = rag(query, use_vector_search=True, vector_field="question_text_vector", vector=query_vector, llm_model=llm_model)
    else:
        answer = rag(query, use_vector_search=False, llm_model=llm_model)

    # Display the answer
    st.subheader("Generated Answer")
    st.write(answer)
