import streamlit as st
from elasticsearch import Elasticsearch
from groq import Groq
from sentence_transformers import SentenceTransformer
import json
from modular_rag import create_index, index_documents, elastic_search, build_prompt, llm, rag

# Initialize clients
es_client = Elasticsearch('http://localhost:9200')
llm_client = Groq(api_key="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")

# Streamlit app
st.title("Course Q&A Assistant")

# Sidebar for configuration
st.sidebar.header("Search Configuration")

# Search type selection
search_type = st.sidebar.radio("Select Search Type", ["Keyword", "Vector"])

# Select course
course = st.sidebar.selectbox(
    "Select course",
    ['data-engineering-zoomcamp', 'machine-learning-zoomcamp', 'mlops-zoomcamp']
)

# Vector field selection (only show if vector search is selected)
vector_field = None
if search_type == "Vector":
    vector_field = st.sidebar.selectbox(
        "Select Vector Field",
        ["question_vector", "text_vector", "question_text_vector"]
    )

# LLM model selection
llm_models = ["llama3-8b-8192", "llama3-70b-4096", "mixtral-8x7b-32768"]
selected_llm_model = st.sidebar.selectbox("Select LLM Model", llm_models)

# Vector encoding model selection (only show if vector search is selected)
selected_vector_model = None
if search_type == "Vector":
    vector_encoding_models = ["multi-qa-MiniLM-L6-cos-v1", "all-mpnet-base-v2", "all-MiniLM-L6-v2"]
    selected_vector_model = st.sidebar.selectbox("Select Vector Encoding Model", vector_encoding_models)

# Main app area
st.header("Ask a Question")
user_question = st.text_input("Enter your question here:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Searching for answer..."):
            try:
                with open('/home/nkama/LLM_and_RAG_Course/LLM_and_RAG-/personal_assg_project/data/documents-with-ids.json', 'r') as f:
                    documents = json.load(f)
                answer = rag(
                    llm_client,
                    es_client,
                    user_question,
                    course=course,
                    use_vector=(search_type == "Vector"),
                    vector_field=vector_field,
                    llm_model=selected_llm_model
                )
                st.success("Answer found!")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question.")

# Manual answer generation
st.sidebar.header("Manual answer generation")
if st.sidebar.checkbox("Enable Manual answer generation"):
    user_question2 = st.sidebar.text_input("Enter your question for manual generation:")

    if st.sidebar.button("Create Index", key="create_index_button"):
        create_index(es_client, use_vectors=(search_type == "Vector"))
        st.sidebar.success("Index created successfully!")

    if st.sidebar.button("Index Documents", key="index_documents_button"):
        try:
            with open('/home/nkama/LLM_and_RAG_Course/LLM_and_RAG-/personal_assg_project/data/documents-with-ids.json', 'r') as f:
                documents = json.load(f)

            if search_type == "Vector":
                model = SentenceTransformer(selected_vector_model)
                index_documents(es_client, documents, model=model)
            else:
                index_documents(es_client, documents)
            st.sidebar.success("Documents indexed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error indexing documents: {str(e)}")

    if st.sidebar.button("Generate Manual Answer", key="generate_manual_answer_button"):
        if user_question2:
            with st.spinner("Generating manual answer..."):
                try:
                    answer = rag(
                        llm_client,
                        es_client,
                        user_question2,
                        course=course,
                        use_vector=(search_type == "Vector"),
                        vector_field=vector_field,
                        llm_model=selected_llm_model
                    )
                    st.sidebar.success("Manual answer generated!")
                    st.sidebar.write(answer)
                except Exception as e:
                    st.sidebar.error(f"An error occurred: {str(e)}")
        else:
            st.sidebar.warning("Please enter a question for manual generation.")

# Run the app with: streamlit run your_app_name.py
