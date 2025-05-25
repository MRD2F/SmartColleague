import os

from process_files import ProcessText

class CreateChromaCollection:
    def __init__(self, chroma_client, file_path, collection_name, persistent_collection=False, db_path = '', delete_collection=False):
        self.chroma_client = chroma_client
        self.file_path = file_path
        self.collection_name = collection_name
        self.persistent_collection = persistent_collection
        self.db_path = db_path
        self.delete_collection = delete_collection
        #self.collection = self.get_or_create_collection()

    def get_or_create_collection(self):
        if self.delete_collection:
            self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        return self.collection 

    
    def add_to_collection(self):
        #documentation_text = {}
        doc_metadata_names = ["filename", "slide_number", "title"]

        for file_name in os.listdir(self.file_path):
            print(file_name)
            if file_name.endswith(".pdf"):
                print(f"Processing {file_name}...")
                pt = ProcessText(os.path.join(self.file_path, file_name))

                slides = pt.extract_text_from_pdf()
                print(slides)
                break
                #documentation_text[filename] = slides
                for n in range(len(slides)):
                    slide = slides[n] ########## chunk_text(slides[n])
                    print(slide)
                    if False: #slide:

                        self.collection.add(
                        ids=[f"{filename.removesuffix('.pdf')}_slide_{n}"],
                        documents=slide,
                        metadatas=[{doc_metadata_names[0]: filename, 
                                    doc_metadata_names[1]: n,
                                    doc_metadata_names[2]: slides[0].split('Slide 1:')[1]}]
                    )
                else:
                    print("Skipping add: One or more required lists are empty.")
                    print(f"filename: {filename}, slide number: {n}")

        print("✅ All PDFs added to the document collection!")


chroma_client= ''
file_path = '../data/engage_iniziative/'
collection_name = ''
chroma_client= ''
cc = CreateChromaCollection(chroma_client, file_path, collection_name)
cc.add_to_collection()