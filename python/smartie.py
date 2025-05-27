# import pandas as pd
# import re
import os
import chromadb
from IPython.display import Markdown, display

from google import genai
from google.genai import types
import pdfplumber
import textwrap
import gradio as gr

from dotenv import load_dotenv
import os

class pippo:
    def __init__(self, name):
        self.name = name

class QueryCollection:
    def __init__(self, chroma_db_path, doc_collection_name):#cv_collection_name, doc_collection_name):
        self.collection_bd_path = chroma_db_path
        #self.collection_name_list = collection_name_list
        self.cv_collection_name = 'cv_collection'
        self.doc_collection_name = doc_collection_name
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        print(self.chroma_client.list_collections())
        
        # self.collections = []

        # self.cv_collection = self.chroma_client.get_collection(cv_collection_name)
        self.doc_collection = self.chroma_client.get_collection(doc_collection_name)

    def get_cb_information(self):
        for col in self.chroma_client.list_collections():
            print(f"Collection Name: {col.name}")
            print(f"Number of Documents: {col.count()}")
            print("-" * 40)

    def execute_query(self, cv_collection_name: str, user_query : str) -> dict:
        """Execute a potentially read-only query and return the results."""
        collection = self.chroma_client.get_collection(name=cv_collection_name)
        n_results=10
        results = collection.query(
            query_texts=[user_query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        return results
    
    def compute_rag(self, collection_name: str, query : str, n_results : int, include : list) -> str:
        #Retrieve relevant docs from vector DB
        
        collection = self.chroma_client.get_collection(name=collection_name)

        retrieved_snippets = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=include
        )

        #Combine retrieved snippets into prompt
        context_text = "\n".join(retrieved_snippets["documents"][0])
        prompt = f"Use the following company info to answer the question:\n{context_text}\n\nQuestion: {query}"

        return prompt
    
    def get_results(self, collection_name : str, query : str) -> dict:
        """Finds the most relevant employee CVs based on the query."""
        collection_name = self.cv_collection_name
        n_results= 10
        collection = self.chroma_client.get_collection(name=collection_name)

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
    
    def get_matching_resumes(self, collection_name : str, query : str) -> dict:
        n_results=10
        collection_name = self.cv_collection_name
        collection = self.chroma_client.get_collection(name=collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def find_similarities(self, collection_name : str, query : str, n_results : int = 10) -> str:
        """Finds the most relevant CVs based on the query."""
        collection_name = 'cv_collection'
        #n_results= 10
        collection = self.chroma_client.get_collection(name=collection_name)

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
    
    def get_employee_resume(self, collection_name : str, employee_id : str) -> dict:
        """Provides information about a specific employee based on their ID."""
        collection = self.chroma_client.get_collection(name=self.cv_collection_name)
        employee_id_info =  collection.get(where={"ID": employee_id})
        employee_id_cv = employee_id_info["documents"][0]
        employee_id_name =  employee_id_info["metadatas"][0]["name"]
        return {'employee_id_cv': employee_id_cv, 
                'employee_id_name': employee_id_name,
                'employee_id': employee_id }

    def show_similarities(self, collection_name : str, query : str) -> str:
        output = self.find_similarities(collection_name, query)  
        print(output)
        return output
    
    def get_matching_documentation(self, query : str) -> dict:
        """Finds the most relevant company documentation based on the query."""
        n_results= 10
        collection = self.chroma_client.get_collection(name=self.doc_collection_name)

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        x = results
        docs_info = {}
        for i in range(len(x['documents'][0])):
            info = x['documents'][0][i]
            id = x['metadatas'][0][i]['file_name']
            docs_info[id] = {
                'document_content': info, 
                'document_name' : id
            }
        return docs_info
    
    def list_company_documentation(self) -> list:
        # Retrieve all documents from the collection
        results = self.doc_collection.get(include=["metadatas"])

        # Extract all 'file_name' fields from metadata
        file_names = [metadata.get('file_name')+'.pdf' for metadata in results['metadatas']]

        return file_names


class GenerateContent:
    def __init__(self, instructions_path, tool_list):
        self.client = self.get_google_client()
        self.instructions_path = instructions_path
        self.instructions = self.load_instructions(instructions_path)
        self.tool_list = tool_list

    def get_google_client(self):
        load_dotenv()  # Load .env file
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY in .env") 
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        return self.client
    
    def load_instructions(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
        

    def generate_content(self, user_query):
        """Generates content using the Google GenAI API."""
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=[self.instructions],
                tools=self.tool_list
            ),
        )
        display(Markdown(response.text))

# instructions_path = '../cofig/genai_instructions.txt'
# GenerateContent(instructions_path, '')
# db_path='../data/chromaDB'
# qc = QueryCollection(chroma_db_path=db_path, doc_collection_name="doc_collection_new")
# print(qc.list_company_documentation())