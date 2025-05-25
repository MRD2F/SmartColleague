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
            slides = []
            for page in doc:
                text = page.get_text()
                if text:
                    text = text.strip() 
                    text = self.clean_text(text) if apply_cleaning else text
                    slides.append(text)
            slides = [f"Slide {i+1}: {text}" for i, text in enumerate(slides)]
            return slides
        except Exception as e:
            print(f"Error reading PDF: {e}")
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
    

