import os
import chromadb
import textwrap
from process_text import ProcessText

class ChromaCollection:
    def __init__(self, persistent_collection=False,
                  db_path = '', delete_collection=False):
        self.persistent_collection = persistent_collection
        if self.persistent_collection:
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        else:
            self.chroma_client = chromadb.Client()

        self.db_path = db_path
        self.delete_collection = delete_collection

    def collection_exists(self, collection_name):
        """Check if the collection already exists."""
        collections = self.chroma_client.list_collections()
        if not any(col.name == collection_name for col in collections):
            print(f"Collection '{collection_name}' does not exist.")
            
        return any(col.name == collection_name for col in collections)

    def get_create_collection(self, collection_name):
        collection_exists = self.collection_exists(collection_name)
        if self.delete_collection and collection_exists:
            self.chroma_client.delete_collection(name=collection_name)
        if not collection_exists:
            print(f"Creating collection '{collection_name}'...")
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        return collection 
    

    def add_to_collection(self, collection_name, file_path, n_files_to_add=500):
        doc_metadata_names = ["file_name", "page_number", "title"]

        ##### Check if the collection exists, if not create one #####
        if not self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' does not exist. Creating a new collection.")
        collection = self.get_create_collection(collection_name)

        ##############################################################
        n_files_added=0
        print(f"There are {len(os.listdir(file_path))} files in the directory {file_path}.")
        for file_name in os.listdir(file_path):
            adding_file = True

            if not file_name.endswith(".pdf"):
                print(f"Skipping {file_name} as it is not a PDF")
                adding_file = False
            else:
                print(f"Processing {file_name}...")                

                #Read from PDF File and clean text
                pt = ProcessText(os.path.join(file_path, file_name))
                slides = pt.extract_text_from_pdf()
                ###################################

                if len(slides) == 0:
                    print(f"No pages found in {file_name}. Skipping...")
                    adding_file = False
                    continue
                else:
                    print(f"Found {len(slides)} pages in {file_name}. Processing...")
                for n in range(len(slides)):
                    slide = slides[n] 
                    chunks = pt.chunk_text([slide])  
                    #chunks = pt.basic_chunk_text(slide)

                    if chunks and slide:
                        collection.add(
                            ids=[f"{file_name.removesuffix('.pdf')}_slide_{n}"],
                            documents=slide,
                            metadatas=[{doc_metadata_names[0]: file_name, 
                                        doc_metadata_names[1]: n,
                                        doc_metadata_names[2]: slides[0].split('Slide 1:')[1]}]
                    )
                    else:
                        adding_file = False
                        print("Skipping add: One or more required lists are empty.")
                        print(f"file_name: {file_name}, page number: {n}")
            if adding_file:
                n_files_added+=1
                print(f"Added {n_files_added}/{len(os.listdir(file_path))} files to the collection.")
                # Check if we have reached the limit of files to add
            if n_files_added >= n_files_to_add:
                print(f"Reached the limit of {n_files_to_add} files to add. Stopping.")
                break
        print("✅ All PDFs added to the document collection!")


# chroma_client= ''
# file_path = '../data/engage_iniziative/'
# collection_name = ''
# chroma_client= ''
# cc = ChromaCollection(chroma_client, file_path, collection_name)
#cc.add_to_collection()

# file_path = '../data/company_documentation/'
# doc_collection_name = 'doc_collection'

# persistent_collection = True
# collection_created=False
# delete_collection = True

# vector_db_path='../data/chromaDB'
# os.makedirs(vector_db_path, exist_ok=True)

# cc = ChromaCollection(file_path, doc_collection_name, collection_created, persistent_collection, vector_db_path ,delete_collection)

# cc.add_to_collection()

# file_path = '../data/company_documentation/campione'
# db_path='../data/chromaDB'
# delete_collection = False
# persistent_collection = False
# collection_name = 'doc_collection_test'
# cc = ChromaCollection(persistent_collection, db_path, delete_collection)
# cc.add_to_collection_new(collection_name, file_path, n_files_to_add=3)
# cc.collection_exists(collection_name)
# print(cc.chroma_client.list_collections())
# cc.get_create_collection(collection_name)
# print(cc.chroma_client.list_collections())

