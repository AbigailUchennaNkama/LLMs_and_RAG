import argparse
from tqdm import tqdm
from elasticsearch import Elasticsearch

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
                "course": {"type": "keyword"}
            }
        }
    }

    es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully.")

def index_documents(es_client, index_name, documents):
    for doc in tqdm(documents, desc="Indexing documents"):
        try:
            es_client.index(index=index_name, document=doc)
        except Exception as e:
            print(f"Error indexing document {doc['id']}: {e}")

def elastic_search(es_client, index_name, query, course):
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
    result_docs = [hit['_source'] for hit in response['hits']['hits']]

    return result_docs

def main(documents, query, course):
    # Initialize Elasticsearch client
    es_client = Elasticsearch('http://localhost:9200')

    # Define index name
    index_name = "course-questions"

    # Create the index
    create_index(es_client, index_name)

    # Index the documents
    index_documents(es_client, index_name, documents)

    # Perform a keyword search
    search_results = elastic_search(es_client, index_name, query, course)
    print("Search results:", search_results)

if __name__ == "__main__":
    # Example usage
    documents = [
        {
            "id": "1",
            "question": "What is the capital of France?",
            "text": "The capital of France is Paris.",
            "section": "Geography",
            "course": "world-history"
        },
        {
            "id": "2",
            "question": "Explain the theory of relativity.",
            "text": "The theory of relativity was developed by Albert Einstein.",
            "section": "Physics",
            "course": "modern-physics"
        }
        # Add more documents as needed
    ]

    query = "capital of France"
    course = "world-history"

    main(documents, query, course)
