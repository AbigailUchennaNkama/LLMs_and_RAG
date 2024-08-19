from elasticsearch import Elasticsearch
from groq import Groq
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer
from elasticsearch.helpers import bulk

# Initialize clients
es_client = Elasticsearch('http://localhost:9200')
llm_client = Groq(api_key="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")

def create_index(es_client, index_name="course-questions", use_vectors=True):
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
                "id": {"type": "keyword"}
            }
        }
    }

    if use_vectors:
        vector_fields = {
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
            }
        }
        index_settings["mappings"]["properties"].update(vector_fields)

    es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully.")

def index_documents(es_client, documents, index_name="course-questions", model=None):
    def generate_actions():
        for doc in documents:
            if model:
                question = doc['question']
                text = doc['text']
                qt = question + ' ' + text
                doc['question_vector'] = model.encode(question).tolist()
                doc['text_vector'] = model.encode(text).tolist()
                doc['question_text_vector'] = model.encode(qt).tolist()

            yield {
                "_index": index_name,
                "_id": doc['id'],
                "_source": doc
            }

    success, failed = bulk(es_client, generate_actions(), stats_only=True, raise_on_error=False)

    print(f"Indexed {success} documents successfully.")
    if failed:
        print(f"Failed to index {failed} documents.")

def elastic_search(es_client, query, index_name="course-questions", use_vector=False, vector_field=None, course="data-engineering-zoomcamp"):
    if use_vector and vector_field:
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
        vector = model.encode(query).tolist()
        knn = {
            "field": vector_field,
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
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def rag(llm_client, es_client, query, course="data-engineering-zoomcamp", use_vector=False, vector_field=None, llm_model="llama3-8b-8192"):
    search_results = elastic_search(es_client, query, use_vector=use_vector, vector_field='question_text_vector', course=course)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, llm_client, llm_model)
    return answer

# Example usage:
# create_index(es_client, use_vectors=True)
#
# with open('documents.json', 'r') as f:
#     documents = json.load(f)
#
# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# index_documents(es_client, documents, model=model)
#
# query = "What is data engineering?"
# result = rag(llm_client, es_client, query, use_vector=True, vector_field="question_text_vector")
# print(result)
