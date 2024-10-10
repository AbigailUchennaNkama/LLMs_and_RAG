import os
from tqdm.auto import tqdm
from dotenv import load_dotenv
import pandas as pd
from litellm import completion
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from elasticsearch.helpers import bulk

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize clients
es_client = Elasticsearch('http://localhost:9200')

def create_index(es_client, index_name='interview_qa', use_vectors=True):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "course": {"type": "keyword"},
                # Added vector fields directly to the main properties
                "question_vector": {"type": "dense_vector", "dims": 384} if use_vectors else None,
                "answer_vector": {"type": "dense_vector", "dims": 384} if use_vectors else None,
                "question_answer_vector": {"type": "dense_vector", "dims": 384} if use_vectors else None,
            }
        }
    }

    # Delete the existing index if it exists
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    # Create the new index
    es_client.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully.")

def index_documents(es_client, documents, index_name='interview_qa', model=None):
    def generate_actions():
        for doc in tqdm(documents, desc="Indexing documents"):
            # Ensure model is provided
            if model is None:
                raise ValueError("Model must be provided for encoding.")
                
            # Encode the text fields
            question_vector = model.encode(doc['question']).tolist()
            answer_vector = model.encode(doc['answer']).tolist()
            question_answer_vector = model.encode(doc['question'] + " " + doc['answer']).tolist()

            # Prepare the document for indexing
            index_doc = {
                "question": doc['question'],
                "answer": doc['answer'],
                "question_vector": question_vector,
                "answer_vector": answer_vector,
                "question_answer_vector": question_answer_vector,
                "doc_id": doc.get('doc_id', None)  # Use None if 'doc_id' is not present
            }

            yield {
                "_index": index_name,
                "_id": doc['doc_id'],
                "_source": index_doc  # Use index_doc instead of doc
            }

    success, failed = bulk(es_client, generate_actions(), stats_only=True, raise_on_error=False)

    print(f"Indexed {success} documents successfully.")
    if failed:
        print(f"Failed to index {failed} documents.")

def elastic_search(es_client, query, index_name='interview_qa', use_vector=False,
                   vector_field=None, course=None, model_name='all-MiniLM-L6-v2'):
    if use_vector and vector_field:
        model = SentenceTransformer(model_name)
        vector = model.encode(query).tolist()
        knn = {
            "field": vector_field,
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10000,
            "filter": [{"term": {"course": course}}] if course else []
        }
        
        search_query = {
            "knn": knn,
            "_source": ["answer", "question", "course", "doc_id"]
        }
    else:
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "answer", "course"],
                            "type": "best_fields"
                        }
                    }
                }
            }
        }

    response = es_client.search(index=index_name, body=search_query)
    return [hit['_source'] for hit in response['hits']['hits']]

def build_prompt(query, search_results):
    prompt_template = """
    You're a Technical Interview Assistant. Your role is to Assist candidates 
    preparing for interviews by providing detailed 
    explanations, sample answers, and coding examples for Data Science, 
    Python, and SQL-related interview questions. 
    Use only the facts from the CONTEXT when answering the QUESTION. Do not answer from
    own knowledge. If you do not find an appropriate answer to the query, just return the text
    "No suitable answer found"

    QUESTION: {question}

    CONTEXT: 
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context += f"question: {doc['question']}\nanswer: {doc['answer']}\ncourse: {doc['course']}\n\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm_response(prompt, model="groq/llama3-8b-8192"):
    response = completion(
        model=model, 
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

def rag(llm_client, es_client, query, course=None,
        use_vector=False, vector_field=None, llm_model="llama3-8b-8192"):
    search_results = elastic_search(es_client, query, use_vector=use_vector, vector_field=vector_field, course=course)
    prompt = build_prompt(query, search_results)
    answer = llm_response(prompt, llm_model)  # Corrected to call llm_response
    return answer

def rag_system(documents, llm_client, es_client, query, course=None,
               use_vector=False, vector_field=None, llm_model="llama3-8b-8192"):
    create_index(es_client, index_name="course-questions", use_vectors=True)
    model = SentenceTransformer(llm_model)  # Initialize the model here
    index_documents(es_client, documents, index_name="course-questions", model=model)  # Pass the model
    search_results = elastic_search(es_client, query, use_vector=use_vector, vector_field=vector_field, course=course)
    prompt = build_prompt(query, search_results)
    answer = llm_response(prompt, llm_model)  # Corrected to call llm_response
    return answer