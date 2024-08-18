from elasticsearch import Elasticsearch
from groq import Groq
from tqdm import tqdm
import json
# Initialize clients
es_client = Elasticsearch('http://localhost:9200')
client = Groq()

# Keyword Search Functions
def create_keyword_index(es_client, index_name):
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
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully.")

def index_keyword_documents(es_client, index_name, documents):
    for doc in tqdm(documents, desc="Indexing documents"):
        try:
            es_client.index(index=index_name, document=doc)
        except Exception as e:
            print(f"Error indexing document {doc['id']}: {e}")

def keyword_elastic_search(es_client, index_name, query):
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
    response = es_client.search(index=index_name, body=search_query)
    return [hit['_source'] for hit in response['hits']['hits']]

# Vector Search Functions
def create_vector_index(es_client, index_name):
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
    es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Vector index '{index_name}' created successfully.")

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
            print(f"Error indexing document {doc['id']}: {e}")

def vector_elastic_search(es_client, index_name, field, vector, course):
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
    es_results = es_client.search(index=index_name, body=search_query)
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

# RAG Function
def rag(query, index_name="course-questions", use_vector_search=False, vector_field=None, vector=None, llm_model="llama3-8b-8192"):
    if use_vector_search and vector_field and vector is not None:
        search_results = vector_elastic_search(es_client, index_name, vector_field, vector, "data-engineering-zoomcamp")
    else:
        search_results = keyword_elastic_search(es_client, index_name, query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, client, llm_model)
    return answer

# Example usage
with open('documents-with-ids.json', 'rt') as f_in:
    documents = json.load(f_in)
# index_name = "course-questions"
# create_keyword_index(es_client, index_name)
# create_vector_index(es_client, index_name)
# query = "What is data engineering?"
# result = rag(query)
