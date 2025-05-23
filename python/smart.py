import pandas as pd
import numpy as np  
import re
import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from IPython.display import Markdown, display

from google import genai
from google.genai import types
import pdfplumber
import textwrap
import warnings
import gradio as gr

from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()  # Load .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env") 

client = genai.Client(api_key=GOOGLE_API_KEY)


read_doc_pdfs=False
doc_create_new_collection=False

###########################################
read_cv_pdfs=False
cv_create_new_collection=False
cv_text_from_pdf_path = '../data/cvs_from_pdf'
cv_pdf_path = '../../curriculum_vitae_data/pdf'

#############################################

#chroma_client = chromadb.Client()
vector_db_path='../data/chromaDB'
os.makedirs(vector_db_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=vector_db_path)

doc_collection_name='doc_collection'
doc_collection = chroma_client.get_or_create_collection(name=doc_collection_name)

cv_collection_name='cv_collection'
cv_collection = chroma_client.get_or_create_collection(name=cv_collection_name)


############ GENERAL DATA PROCESSING FUNCTIONS #############
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text(use_cropbox=False) + "\n"
    return text

def chunk_text(text, max_length=500):
    return textwrap.wrap(text, max_length)

############ QUERY FUNCTIONS ####################################

def compute_rag(collection_name: str, query : str, n_results : int, include : list) -> str:
    #Retrieve relevant docs from vector DB
    
    collection = chroma_client.get_collection(name=collection_name)

    retrieved_snippets = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=include
    )

    #Combine retrieved snippets into prompt
    context_text = "\n".join(retrieved_snippets["documents"][0])
    prompt = f"Use the following company info to answer the question:\n{context_text}\n\nQuestion: {query}"

    return prompt
    
def get_results(collection_name : str, query : str) -> dict:
    """Finds the most relevant employee CVs based on the query."""
    n_results= 10
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    x = results
    employees_info = {}
    for i in range(len(x['documents'][0])):
        info = x['documents'][0][i]
        id = x['ids'][0][i]
        name = x['metadatas'][0][i]['name']
        employees_info[id] = {
            'employee_name': name,
            'info': info, 
            'employee_id' : id
        }
    return employees_info

def get_matching_resumes(collection_name : str, query : str) -> dict:
    n_results=10
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

def find_similarities(collection_name : str, query : str) -> str:
    """Finds the most relevant CVs based on the query."""
    n_results= 10
    collection = chroma_client.get_collection(name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    output = f"Found {len(results['ids'][0])} similar employees for query: '{query}'.\n"
    output += "-" * 80 + "\n"
    for i, (doc_id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity_score = (1 - distance) * 100
        #output += f"\n{i+1}. Recipe Name: {metadata.get('name', 'Unnamed')}\n"
        output += f"   Similarity: {similarity_score:.2f}%\n"
        output += f"   Employee ID: {doc_id}\n"
        output += f"   Employee Name: {metadata.get('name', 'Unnamed')}\n"
        for key, value in metadata.items():
            output += f"   {key.replace('_', ' ').title()}: {value}\n"
        output += f"   resume: {doc}\n"  # Include the full document 
        output += "-" * 80 + "\n"
    return output

def get_employee_info(collection_name : str, employee_id : str) -> dict:
    """Provides information about a specific employee based on their ID."""
    collection = chroma_client.get_collection(name=collection_name)
    employee_id_info =  collection.get(where={"ID": employee_id})
    employee_id_cv = employee_id_info["documents"][0]
    employee_id_name =  employee_id_info["metadatas"][0]["name"]
    return {'employee_id_cv': employee_id_cv, 
            'employee_id_name': employee_id_name,
            'employee_id': employee_id }

def show_similarities(collection_name : str, query : str) -> str:
    #results = get_matching_resumes(collection_name, query)

    output = find_similarities(collection_name, query)  
    print(output)
    return output

def get_matching_documentation(query : str) -> dict:
    n_results=5
    collection = chroma_client.get_collection(name=doc_collection_name)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return results

def get_matching_documentation(query : str) -> dict:
    """Finds the most relevant company documentation based on the query."""
    n_results= 10
    collection = chroma_client.get_collection(name=doc_collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    x = results
    docs_info = {}
    for i in range(len(x['documents'][0])):
        info = x['documents'][0][i]
        id = x['metadatas'][0][i]['source']
        docs_info[id] = {
            'document_content': info, 
            'document_name' : id
        }
    return docs_info

############ GenAI FUNCTIONS ####################################


def get_instructions():
    instructions = """You are a helpful employee of a company named Engage Group, you know all its employees, their CVs and capabilities. You want to help your employees colleghes to conect so they can help each other, improving their comunication, knowleadge about each others and their difference proyects, with the main goal of improving the work enviroment and productivity.
        You want to help your employees colleghes getting thought all the company documentation, manuals, instruction for requests and all information regarding HR.
    
    You have access to special tools that let you:
    - Retrieve resume CV of all the employees, stored in the collection named cv_collection.
    - Query the resume information to match the request of the user, based on the similarity to the user query.
    - Access to all the Engage documentation, stored in the collection named doc_collection
    - Access to the collection of the Employees CV (cv_collection) and the company documentation (doc_collection)

    ---

    Available Tools:
    - **get_results(collection_name : str, query : str)**  
        Finds the most relevant CVs in the system collection (cv_collection) based on the query, returns a dictionary with the cv_collection results after being quered based on the user query.
    - **find_similarities(collection_name : str, query : str)**  
        Finds the most relevant CVs in the system collection (cv_collection) based on the query, returns a string with the most relevant employee similarity score and summary of their resume / CV.
    - **get_employee_info(collection_name : str, employee_id : str)**  
        Provides information about a specific employee based on the employee ID, returns a dictionary with the employee: CV, name and ID.
    - **get_matching_documentation(query : str)**
        Finds the most relevant company documentation in the system collection based on the query, returns a dictionary with the documentation collection.

    ---

        How to respond:

        1. Identify the user's intent, and brevely summarize the problem at the beginning.
        2. Use the **most relevant tool(s)** to get the potential employees which skills match the user query.
        3. Use `find_similarities` to provide detail information about the possible matches between the user query and employees skills you found, include a short summary.
        4. Use `get_employee_info` to get the CV of the employee ID you found, and include it in the response.
        5. Use `get_matching_documentation` only for company related querys, to get the all the documentation information related to the user request, and include it in the response.
        6. Present results clearly using Markdown formatting.
        7. Include a clear note explaining why you think the results are relevant, and how the experience of the ID can help the user.
        8. After presenting the results, Summarize the keywords skills of the empoyee IDs you found

        ---

        Examples:

        - **User:** "Tell me which empoyee ID can help me with a problem of related to machine learning."  
        **LLM Calls:**  
            1. `find_similarities(collection_name="cv_collection", query="Tell me which empoyee ID can help me with a problem of related to machine learning.")`  
        - **User:** "Tell me all the information of the empoyee ID 11813872."  
        **LLM Calls:**  
            1. `get_employee_info(collection_name="cv_collection", employee_id="11813872")`  
        - **User:** "Explain how its the appraisal cycle for a employee at Engage."  
        **LLM Calls:**  
            1. `get_matching_documentation(query="Explain how its the appraisal cycle for a employee at Engage")`  

        ---

        Be polite, smart, helpful, and accurate. Don't guess data—use the tools! and remember you are talking like a colleague of the user, so be friendly and helpful.
        Regarding technical bureaucratic matters the user may be confused with all the documentation, so be gentle, kind and provide simple steps and explinations. 

        """



    return instructions

def generate_content(user_query, instructions, tools):
    """Generates content using the Google GenAI API."""
    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=[instructions],
            tools=tools
        ),
    )
    display(Markdown(response.text))

tools = [get_results, find_similarities, get_employee_info, get_matching_documentation] 

instructions = get_instructions()

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-001",
    system_instruction=instructions,
    tools=tools 
)
chat = model.start_chat(history=[], enable_automatic_function_calling=True)

print(chat.send_message('hi').text)