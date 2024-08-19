from elasticsearch import Elasticsearch
from groq import Groq
from tqdm import tqdm
import json

from sentence_transformers import SentenceTransformer
#model = SentenceTransformer("all-mpnet-base-v2")
# Initialize clients
es_client = Elasticsearch('http://localhost:9200')
llm_client = Groq(api_key="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")

# Keyword Search Functions
def create_keyword_index(es_client):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }
    index_name = "course-questions"
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully.")

def index_keyword_documents(es_client, index_name, documents):
    for doc in tqdm(documents, desc="Indexing documents"):
        try:
            es_client.index(index=index_name, document=doc)
        except Exception as e:
            print(f"Error indexing document {doc['id']}: {e}")

def keyword_elastic_search(es_client, query):
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
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    index_name = "course-questions"
    response = es_client.search(index=index_name, body=search_query)
    return [hit['_source'] for hit in response['hits']['hits']]

# Vector Search Functions
def create_vector_index(es_client):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
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
                    "similarity": "cosine"
                },
                "text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "question_text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
            }
        }
    }
    index_name = "course-questions"
    es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Vector index '{index_name}' created successfully.")

def encode_and_index_documents(es_client, model, documents):
    for doc in tqdm(documents, desc="Encoding documents"):
        question = doc['question']
        text = doc['text']
        qt = question + ' ' + text
        doc['question_vector'] = model.encode(question)
        doc['text_vector'] = model.encode(text)
        doc['question_text_vector'] = model.encode(qt)

    for doc in tqdm(documents, desc="Indexing documents"):
        try:
            es_client.index(index="course-questions", document=doc)
        except Exception as e:
            print(f"Error indexing document {doc['id']}: {e}")

def vector_elastic_search(es_client, field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
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
    es_results = es_client.search(index="course-questions", body=search_query)
    return [hit['_source'] for hit in es_results['hits']['hits']]

# Prompt Building and LLM Functions
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
    print(response.choices[0].message.content)
    return response.choices[0].message.content

# Separated RAG Functions
def rag_with_vector_search(llm_client, es_client, query, course,vector_field, llm_model, model_encode='multi-qa-MiniLM-L6-cos-v1'):
    model = SentenceTransformer(model_encode)
    vector = model.encode(query).tolist()
    search_results = vector_elastic_search(es_client, vector_field, vector, course)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, llm_client, llm_model)
    return answer

def rag_with_keyword_search(llm_client, es_client, query, llm_model="llama3-8b-8192"):
    search_results = keyword_elastic_search(es_client, "course-questions", query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, llm_client, llm_model)
    return answer
