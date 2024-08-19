import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import json

def create_index(es_client, index_name):
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

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully.")

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

def elastic_search_knn(es_client, index_name, field, vector, course):
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
    result_docs = [hit['_source'] for hit in es_results['hits']['hits']]

    return result_docs

def main(documents):
    # Initialize Elasticsearch client
    es_client = Elasticsearch()

    # Define index name
    index_name = "course-questions"

    # Create the index
    create_index(es_client, index_name)

    # Load the model
    model = SentenceTransformer("all-mpnet-base-v2")

    # Encode and index documents
    encode_and_index_documents(es_client, index_name, model, documents)

    # Example of performing a KNN search
    # vector_to_search = model.encode("sample question to search")
    # search_results = elastic_search_knn(es_client, index_name, "question_vector", vector_to_search, "course-name")
    # print(search_results)

if __name__ == "__main__":
    # Example usage
    with open('/home/nkama/LLM_and_RAG_Course/LLM_and_RAG-/data/documents-with-ids.json', 'rt') as f_in:
        documents = json.load(f_in)
    main(documents)
