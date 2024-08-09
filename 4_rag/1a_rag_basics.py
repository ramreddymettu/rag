import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Check if the Chroma vector store already exists
current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_odyssey")
# file_path = os.path.join(current_dir, "books", "us_bill_of_rights.txt")
# persistent_directory = os.path.join(current_dir, "db", "chroma_us_bill_of_rights")
print({"current_dir": current_dir, "file_path": file_path, "persistent_directory": persistent_directory})

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    # Read the test file exists
    loader = TextLoader(file_path=file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n{docs[0].page_content}\n")


    # Crete embeddins
    embeddins = OllamaEmbeddings(
        model="llama3.1"
    )

    print("\n--- Finished creating embeddins ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        documents=docs, embedding=embeddins,persist_directory=persistent_directory
    )

    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")