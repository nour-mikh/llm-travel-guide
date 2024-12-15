# Import required modules
import os  # For file and directory operations

# Import vector database and embedding modules from langchain
from langchain_chroma import Chroma  # Vector database for storing embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Google's embedding model

# Import document loader for PDFs
from langchain_community.document_loaders import PyPDFLoader  # Loads and parses PDF files

# Google API key for authentication
api_key = ""

# Initialize the embedding model with Google's embedding-001
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Create/Load a Chroma vector database
vector_store = Chroma(
    collection_name = "embeddings",  # Name of the collection to store vectors
    embedding_function = embeddings,  # Function to generate embeddings
    persist_directory = "./vector_db",  # Directory to save the database
    collection_metadata = {"hnsw:space": "cosine"}  # Use cosine similarity for vector comparisons
)

# Directory containing PDF files to process
directory_path = "C:\\Users\\samer.saabjr\\Documents\\LAU\\My Courses\\LLMs\\Assessments\\Project\\Directory"

# Loop through all PDF files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a PDF
    if filename.endswith('.pdf'):
        # Create full file path by joining directory and filename
        file_path = os.path.join(directory_path, filename)
        # Create a PDF loader for the current file
        loader = PyPDFLoader(file_path)
        # Load and parse the PDF into documents
        documents = loader.load()
        # Add the documents to the vector store with their embeddings
        vector_store.add_documents(documents)

# Example similarity search (commented out)
# result = vector_store.similarity_search(query = "matrix multiplication", k = 5)  # Search for 5 most similar documents
# for doc in result:
#     print(doc.page_content)  # Print the content of each similar document
