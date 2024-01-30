#Streamlit, a framework for building interactive web applications.
#It provides functions for creating UIs, displaying data, and handling user inputs.
import streamlit as st
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
#This module provides a way to interact with the operating system, such as accessing environment variables, working with files
#and directories, executing shell commands, etc
import pypdf
import os

#By st.set_page_config(), you can customize the appearance of your Streamlit application's web page
st.set_page_config(page_title="Advice Seeking", page_icon=":robot:")
st.header("Good Evening...part-time students, this app help you to understand the content of any readings")
st.session_state.prompt_history = []

if "openai_key" not in st.session_state:
    #with st.form("API key"):
        #key = st.text_input("OpenAI Key", value="", type="password")
        #if st.form_submit_button("Submit"):
   st.session_state.openai_key = os.environ["OPENAI_API_KEY"]
            #st.success('Saved API key for this session.')

# An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. 
# Small distances suggest high relatedness and large distances suggest low relatedness.
# Generate Text Embedding using different LLM
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings

#FAISS is an open-source library developed by Facebook AI Research for efficient similarity search and 
#clustering of large-scale datasets, particularly with high-dimensional vectors. 
#It provides optimized indexing structures and algorithms for tasks like nearest neighbor search and recommendation systems.
from langchain.vectorstores import FAISS

#The below snippet helps us to import structured pdf file data for our tasks
from langchain.document_loaders import PyPDFDirectoryLoader

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

# Assigning the data inside the pdf to our variable here
# Passing the directory to the 'load_docs' function or Get the doc
  
if st.button("database docs"): 

    directory = 'data'
    documents = load_docs(directory)
    st.write(len(documents))
    docs = split_docs(documents)
    st.write("Approx number of token", len(docs))
    
if st.button ("upload docs")
    documents = st.file_uploader("Upload documents here, only PDF file allowed", type=["pdf"], accept_multiple_files=True)
    st.write(len(documents))
    docs = split_docs(documents)
    st.write("Approx number of token", len(docs))


# Initialize the OpenAIEmbeddings object
# Using OpenAI specified models
#embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")  
# OR Using Hugging Face LLM for creating Embeddings for documents/Text
#from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = OpenAIEmbeddings()
  
#Store and Index vector space
db = FAISS.from_documents(docs, embeddings)

# LLM Q&A Code
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")

# This function will transform the question that we raise into input text to search relevant docs
def get_text():
    input_text = st.text_input("For example: Summarise the provincial organization? ", key = input)
    return input_text

#This function will help us in fetching the top k relevent documents from our vector store - Pinecone
def get_similiar_docs(query, k=2):
    similar_docs = db.similarity_search(query, k=k)
    return similar_docs

# This function will help us get the answer from the relevant docs matching input text
def get_answer(query):
  relevant_docs = get_similiar_docs(query)
  print(relevant_docs)
  response = chain.run(input_documents=relevant_docs, question=query)
  return response

if "sessionMessages" not in st.session_state:
     st.session_state.sessionMessages = [
        SystemMessage(content=" It is wished we are helpful assistants.")
    ]
input_text = get_text()
submit = st.button('Generate')  

if submit:
    response = get_answer(input_text)
    st.subheader("Answer:")
    st.write(response,key= 1)
    if response is not None:
        st.session_state.prompt_history.append(input_text + "  Answer: " + response)

st.subheader("Prompt history:")
st.write(st.session_state.prompt_history)

if st.button("Clear"):
    st.session_state.prompt_history = []
    docs = None