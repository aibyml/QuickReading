#Streamlit, a framework for building interactive web applications.
#It provides functions for creating UIs, displaying data, and handling user inputs.
import streamlit as st
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfplumber
#This module provides a way to interact with the operating system, such as accessing environment variables, working with files
#and directories, executing shell commands, etc
import pypdf
import os
from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
from util import *

# An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. 
# Small distances suggest high relatedness and large distances suggest low relatedness.
# Generate Text Embedding using different LLM
##from langchain.embeddings import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings

#FAISS is an open-source library developed by Facebook AI Research for efficient similarity search and 
#clustering of large-scale datasets, particularly with high-dimensional vectors. 
#It provides optimized indexing structures and algorithms for tasks like nearest neighbor search and recommendation systems.
##from langchain.vectorstores import FAISS

#By st.set_page_config(), you can customize the appearance of your Streamlit application's web page
st.set_page_config(page_title="Learning", page_icon=":robot:")
st.header("Hi...students, this app help you to write your (research) paper, ask me how to use this app to speed up the learning when needed")

if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [
        SystemMessage(content= "It is wished we are helpful assistants.")
    ]

if 'generated' not in st.session_state:
    st.session_state["generated"] = []
    st.session_state.db = None 

if 'input_text' not in st.session_state:
    st.session_state["input_text"] = []

if "openai_key" not in st.session_state:
    #with st.form("API key"):
    #key = st.text_input("OpenAI Key", value="", type="password")
    #if st.form_submit_button("Submit"):
    st.session_state.openai_key = os.environ["OPENAI_API_KEY"]
    #st.success('Saved API key for this session.')

##from langchain.llms import OpenAI
##from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFDirectoryLoader

#The below snippet helps us to import structured pdf file data for our tasks
def load_docs(directory):
    for filename in os.listdir(directory):
        # Loads PDF files available in a directory with pypdf
        if filename.endswith('.pdf'):
            return load_docspdf(directory)
        # Passing the directory to the 'load_docs' function
        elif filename.endswith('.xlsx'):
            return load_docsexcel(directory)
        else:
            print(f"Unsupported file format: {filename}")

def load_docspdf(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            loader = PyPDFDirectoryLoader(directory)
            documents = loader.load()
    return documents

#This function will split the documents into chunks
def split_docs(documents, chunk_size=3000, chunk_overlap=20):
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      docs = text_splitter.split_documents(documents)
      return docs

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None

# This function will transform the question that we raise into input text to search relevant docs
def get_text():
    input_text = st.text_input("$Prompt$ $responses$ $about$ $content$ $through$ $the$ $AI$ 👇", key = input)
    return input_text

def QA(documents):
    st.session_state.db = split_docs(documents)
    #st.write("Approx number of token", len(docs))
    
    # Assigning the data inside the pdf to our variable here
    db = embed(st.session_state.db)
    
    #llm = OpenAI() 
    #chain = load_qa_chain(llm, chain_type="stuff")
    
    #load the enquiry
    input_text = get_text() 
    
    #work on llm
    submit = st.button("Submit")  
    
    if submit:
        st.session_state.input_text.append(input_text)
        response = get_answer(db, input_text)
        #st.subheader("Answer:")
        st.write(response, key=1)
        if response is not None:
            st.session_state.generated.append(response)
    
    st.subheader("Prompt history:")
    st.write(st.session_state.input_text)
    st.write(st.session_state.generated)
    
    if st.button("Clear"):
        st.session_state.input_text = []
        st.session_state.generated = []
        st.session_state.db = None

directory = 'data'
documents = load_docs(directory)
QA(documents)

if st.session_state.input_text is not None:
    load_docs = st.checkbox("$upload$ $docs$")
    if load_docs:
        docs = None
        docs = st.file_uploader("Upload documents here, only PDF file allowed", type=["pdf"], accept_multiple_files=True)
        if docs is not None:
                #st.write(docs)
                st.session_state.input_text = []
                st.session_state.generated = []
                st.session_state.db = None    
                df = extract_data(docs)
                #st.write("Approx number of token", len(docs))
                if st.button("Learn the content"): 
                        QA(docs)    
    
