import json
import os
import sys
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# LLM Setup
def get_titan_llm():
    llm = Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock, model_kwargs={'maxTokenCount': 4096})
    return llm

def get_llama2_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Prompt Template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Response Generation
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Streamlit App
def main():
    st.set_page_config("📑 Chat with PDFs | AWS Bedrock RAG", page_icon="📖", layout="wide")

    # Custom CSS Styling
    st.markdown("""
        <style>
        .main {
            background: #f4f6f8;
            color: #1c1c1e;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #1a1a1a;
            font-family: 'Poppins', sans-serif;
        }
        .stButton button {
            background: linear-gradient(135deg, #e0ecfc, #c4dfff);
            border: none;
            border-radius: 10px;
            color: #1a1a1a;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            font-weight: 600;
            transition: 0.3s ease-in-out;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #bcd4fc, #9fc8fa);
            color: #000;
            transform: scale(1.02);
        }
        .stTextInput>div>div>input {
            background: #ffffff;
            color: #333333;
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 0.5rem;
            font-size: 1rem;
        }
        .css-1kyxreq {
            padding: 1rem 2rem;
        }
        .stSidebar {
            background: #f0f2f5;
            color: #1c1c1c;
            border-right: 1px solid #e3e3e3;
        }
        .stSidebar h1 {
            font-size: 1.4rem;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("📖 Chat with PDF Documents via AWS Bedrock")

    st.markdown("#### ⚡️ AI-Powered PDF Q&A ")

    user_question = st.text_input("💬 Enter your question here:")

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ RAG Control Panel")
        st.write("---")
        st.markdown("Manage your document vector store and choose your LLM model.")
        if st.button("📑 Update Vector Store"):
            with st.spinner("⚙️ Processing PDFs and updating embeddings..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("✅ Vector store updated successfully!")

    st.write("---")
    col1, col2 = st.columns(2)

    # Titan button
    with col1:
        if st.button("🚀 Generate with Titan LLM"):
            with st.spinner("⏳ Generating response with Titan..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_titan_llm()
                result = get_response_llm(llm, faiss_index, user_question)
                st.markdown(f"#### 📑 Titan LLM Response:\n{result}")
                st.success("✅ Response generated successfully!")

    # Llama button
    with col2:
        if st.button("🦙 Generate with Llama3"):
            with st.spinner("⏳ Generating response with Llama3..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_llama2_llm()
                result = get_response_llm(llm, faiss_index, user_question)
                st.markdown(f"#### 📑 Llama3 Response:\n{result}")
                st.success("✅ Response generated successfully!")

if __name__ == "__main__":
    main()
