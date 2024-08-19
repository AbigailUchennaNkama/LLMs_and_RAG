import streamlit as st
from elasticsearch import Elasticsearch
from groq import Groq
from sentence_transformers import SentenceTransformer
import json
from modular_rag import create_index, index_documents, elastic_search, build_prompt, llm, rag
# Initialize Elasticsearch and LLM clients
es_client = Elasticsearch('http://localhost:9200')
llm_client = Groq(api_key="your_groq_api_key")

# Streamlit App
st.title("Q&A Interface")

# Sidebar for model selection
st.sidebar.title("Configuration")

# Select search type
search_type = st.sidebar.radio("Search Type", ("Keyword Search", "Vector Search"))

# If vector search, select vector field and vector encoding model
if search_type == "Vector Search":
    vector_field = st.sidebar.selectbox(
        "Select Vector Field",
        ("question_vector", "text_vector", "question_text_vector")
    )

    vector_model_name = st.sidebar.text_input(
        "Vector-Encoding Model",
        value="multi-qa-MiniLM-L6-cos-v1"
    )

# Select LLM model
llm_model_name = st.sidebar.text_input(
    "LLM Model",
    value="llama3-8b-8192"
)

# Input for query
query = st.text_input("Enter your question:")

# Button to submit query
if st.button("Get Answer"):
    if query:
        if search_type == "Vector Search":
            model = SentenceTransformer(vector_model_name)
            vector = model.encode(query).tolist()
            search_results = elastic_search(
                es_client, query, use_vector=True,
                vector_field=vector_field,
                course="data-engineering-zoomcamp"
            )
        else:
            search_results = elastic_search(
                es_client, query, use_vector=False,
                course="data-engineering-zoomcamp"
            )

        # Build prompt and get answer
        prompt = build_prompt(query, search_results)
        answer = llm(prompt, llm_client, llm_model_name)

        # Display results
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question to get an answer.")

# Helper functions
def elastic_search(es_client, query, index_name="course-questions", use_vector=False, vector_field=None, course="data-engineering-zoomcamp"):
    if use_vector and vector_field:
        knn = {
            "field": vector_field,
            "query_vector": query,
            "k": 5,
            "num_candidates": 10000,
            "filter": {
                "term": {
                    "course": course
                }
            }
        }
        search_query = {
            "knn": knn,
            "_source": ["text", "section", "question", "course", "id"]
        }
    else:
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": course
                        }
                    }
                }
            }
        }

    response = es_client.search(index=index_name, body=search_query)
    return [hit['_source'] for hit in response['hits']['hits']]

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

def llm(prompt, client, llm_model):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=llm_model
    )
    return response.choices[0].message.content
