#Streamlit, a framework for building interactive web applications.
#It provides functions for creating UIs, displaying data, and handling user inputs.
import streamlit as st
from langchain import HuggingFaceHub

#This module provides a way to interact with the operating system, such as accessing environment variables, working with files
#and directories, executing shell commands, etc
import pypdf
import os

# An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. 
# Small distances suggest high relatedness and large distances suggest low relatedness.
# Generate Text Embedding using different LLM
from langchain.embeddings import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings

#FAISS is an open-source library developed by Facebook AI Research for efficient similarity search and 
#clustering of large-scale datasets, particularly with high-dimensional vectors. 
#It provides optimized indexing structures and algorithms for tasks like nearest neighbor search and recommendation systems.
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

if "openai_key" not in st.session_state:
    #with st.form("API key"):
    #key = st.text_input("OpenAI Key", value="", type="password")
    #if st.form_submit_button("Submit"):
    st.session_state.openai_key = os.environ["OPENAI_API_KEY"]
    #st.success('Saved API key for this session.')

def embed(docs, input_text):
    
    # Initialize the OpenAIEmbeddings object
    # Using OpenAI specified models
    #embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")  
    # OR Using Hugging Face LLM for creating Embeddings for documents/Text
    #from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")    
    # LLM Q&A Code
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    llm = OpenAI() 
    chain = load_qa_chain(llm, chain_type="stuff")

    return response
