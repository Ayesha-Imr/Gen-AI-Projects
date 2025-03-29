import os
import time
import numpy as np
from dotenv import load_dotenv
from pymilvus import MilvusClient, connections, Collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cohere
from docling.document_converter import DocumentConverter


# Load environment variables
load_dotenv()


# Load API keys and endpoints
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
ZILLIZ_ENDPOINT = os.getenv('ZILLIZ_ENDPOINT')
ZILLIZ_TOKEN = os.getenv('ZILLIZ_TOKEN')

# Set up Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Connect to Zilliz Cloud
connections.connect(
    uri=ZILLIZ_ENDPOINT,
    token=ZILLIZ_TOKEN
)

# Load or create the collection
collection_name = "Unencrypted_Data_March_2025"
client = MilvusClient(
    uri=ZILLIZ_ENDPOINT,
    token=ZILLIZ_TOKEN
)
collection = Collection(collection_name)
collection.load()

# Function to extract text from a PDF document through a URL
def extract_pdf_url(document_url):
    # Initialize the DocumentConverter
    converter = DocumentConverter()

    # Convert the document from the URL
    converted_document = converter.convert(document_url)

    # Extract text content from the converted document
    text_content = converted_document.document.export_to_markdown()

    # Return extracted content
    return text_content


# Function to get embeddings using Cohere
def get_embeddings(text):
    response = cohere_client.embed(
        model='embed-english-light-v3.0',
        texts=[text],
        truncate='RIGHT',
        input_type='search_document',
    )
    time.sleep(0.5)
    return response.embeddings[0]

# Function to chunk text
def chunk(text):
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    # Create documents from the text
    texts = text_splitter.create_documents([text])

    # Extract the chunked text from each Document and store it in a list
    chunk_texts = [doc.page_content for doc in texts]

    return chunk_texts


def get_query_embedding(query):
    response = cohere_client.embed(
        model='embed-english-light-v3.0',
        texts=[query],  # Input the query directly as a list
        input_type='search_query',
        truncate='RIGHT',
    )
    return response.embeddings[0]  # Return the embedding

# Function to format search results
def format_search_results(search_results):
    formatted = []
    for i, hit in enumerate(search_results[0], 1):
        entity = hit.entity
        # Access fields directly from the entity
        chunk_text = getattr(entity, 'chunk_text', 'No text available')
        source_url = getattr(entity, 'source_url', 'N/A')
        upload_date = getattr(entity, 'upload_date', 'N/A')
        
        # Clean up newline characters
        cleaned_text = chunk_text.replace('\\n', '\n')
        
        formatted.append(f"""
Result {i} (Similarity Score: {1 - hit.distance:.2f}):
------------------------------------------------------------------
Source URL: {source_url}
Upload Date: {upload_date}

Text Content:
{cleaned_text}
------------------------------------------------------------------
""")
    return "\n".join(formatted)

def perform_single_vector_search(query_embedding):
    search_results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        },
        limit=5,
        output_fields=["chunk_text", "source_url", "upload_date"]
    )
    return format_search_results(search_results)

def process_data(data, source_url):
    processed_data = []
    chunks = chunk(data)
    # Process each chunk
    for chunk_text in chunks:
        embedding = get_embeddings(chunk_text)
        chunk_dict = {
            "source_url": source_url,
            "chunk_text": chunk_text,
            "upload_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vector": embedding  
        }
        processed_data.append(chunk_dict)
    return processed_data

# Function to process data and ingest into the database
def process_and_ingest_data(url):

    # Extract text from the URL
    text_data = extract_pdf_url(url)

    # Process data 
    processed_data = process_data(text_data, url)

    # Insert data into Zilliz Cloud collection
    res = client.insert(
        collection_name=collection_name,
        data=processed_data
    )
    return res



# Function to handle the vector search
def search_query(query):
    # Get query embedding
    query_embedding = get_query_embedding(query)
    # Perform vector search
    search_results = perform_single_vector_search(query_embedding)
    # Return the results
    return search_results

