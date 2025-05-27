import pandas as pd
import re
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
import fitz

class ProcessText:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def extract_text_from_pdf(self, apply_cleaning=True):
        try:
            doc = fitz.open(self.file_path)
            pages = []
            for page in doc:
                text = page.get_text()
                if text:
                    text = text.strip() 
                    text = self.clean_text(text) if apply_cleaning else text
                    pages.append(text)
            pages = [f"Page {i+1}: {text}" for i, text in enumerate(pages)]
            return pages
        except Exception as e:
            print(f"Error in function extract_text_from_pdf caused when reading PDF: {e}")
            return []

        
    def clean_text(self, text):
        """
        Cleans raw text extracted from slides or PDF for use with GenAI models.
        - Joins broken lines
        - Removes repeated headers
        - Normalizes whitespace
        - Adds punctuation if missing at the end of lines
        """
        # Step 1: Replace multiple newlines with a paragraph break
        text = re.sub(r'\n{2,}', '\n\n', text)

        # Step 2: Replace single newlines inside paragraphs with space
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # Step 3: Fix spacing issues
        text = re.sub(r'\s{2,}', ' ', text)

        # Step 4: Add period at end of lines that look like sentence ends
        text = re.sub(r'(?<![.\n])(?<=[a-zA-Z0-9])(\n|$)', '.\n', text)

        #Step 6: Remove extra points
        text = self.clean_pdf_toc(text)
        # text = re.sub(r'\.\s*\.', '.', text)

        # Step 5: Remove repeated headers (assuming headers are all caps)
        text = re.sub(r'(?<!\n)([A-Z\s]{3,})(?=\n)', '', text)
        text = re.sub(r'\n{2,}', '\n', text)
        
        #Standardize bullet points to use '•'."""
        text = re.sub(r'[❑·•\-]+', '•', text)

        # Optional: Capitalize sentences (basic, not perfect)
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)

        return text.strip()
    
    
    def chunk_text(self, text_list, max_chars=2000):
        chunks, chunk = [], ""
        for text in text_list:
            if len(chunk) + len(text) < max_chars:
                chunk += "\n" + text
            else:
                chunks.append(chunk)
                chunk = text
        chunks.append(chunk)
        return chunks
    
    def basic_chunk_text(self, text, max_length=500):
        return textwrap.wrap(text, max_length)
    



    def clean_pdf_toc(self, text):
        lines = re.split(r'\s{2,}|\n', text)  # split by long spaces or newlines
        cleaned_lines = []
        for line in lines:
            # Remove filler dots or long spaces between titles and numbers
            line = re.sub(r'\.{3,}', ' ', line)
            line = re.sub(r'\s{2,}', ' ', line)

            # Optional: try to extract page number at end
            match = re.match(r'(.*?)(\d{1,4})\s*$', line.strip())
            if match:
                title, page = match.groups()
                cleaned_lines.append(f"- {title.strip().capitalize()} (Page {page})")
            else:
                cleaned_lines.append(line.strip().capitalize())
        return '\n'.join(cleaned_lines)
    

