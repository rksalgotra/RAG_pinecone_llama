import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Pinecone

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    print("Generating embeddings...")
    embeddings = OllamaEmbeddings(model="jarvis")

    print("Ingesting into Pinecone...")
    Pinecone.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Finished!")
