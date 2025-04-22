# This script demonstrates a basic Retrieval Question Answering (RQA) system using Langchain and Hugging Face Transformers.
# It loads a small knowledge base from a text file, creates embeddings for the text chunks,
# builds a FAISS vector store for efficient retrieval, and then uses a text-to-text generation model
# (Flan-T5) to answer questions based on the retrieved information.

from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # For creating embeddings using Hugging Face models
from langchain_community.vectorstores import FAISS  # For creating and using a FAISS vector store
from langchain_huggingface.llms import HuggingFacePipeline  # For integrating Hugging Face models as Language Model (LLM)
from langchain.chains import RetrievalQA  # For creating a question-answering chain with retrieval capabilities
from langchain_community.document_loaders import TextLoader  # For loading text documents
from langchain.text_splitter import CharacterTextSplitter  # For splitting text into smaller chunks
from langchain_community.vectorstores.vdms import embedding2bytes  # Utility function for converting embeddings to bytes (though not directly used in this basic example)
from transformers import pipeline  # For easily using pre-trained models from Hugging Face Transformers

# Define a list of documents (knowledge base)
docs = ["Paris is the capital of France.", "The Nile is the longest river."]

# Create a text file named "kb.txt" and write the documents into it
with open("kb.txt", "w") as f:
    for doc in docs:
        f.write(doc + "\n")

# Load the text document from "kb.txt" using TextLoader
loader = TextLoader("kb.txt")
documments = loader.load()

# Create a CharacterTextSplitter to split the loaded document into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
docs_split = text_splitter.split_documents(documments)

# Initialize HuggingFaceEmbeddings with a specific pre-trained model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the split documents and their embeddings
db = FAISS.from_documents(docs_split, embedding)

# Create a Hugging Face pipeline for text-to-text generation using the flan-t5-base model
generator = pipeline("text2text-generation", model="google/flan-t5-base", max_length=200)

# Wrap the Hugging Face pipeline in a Langchain HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=generator)

# Create a RetrievalQA chain
# - llm: The language model to use for generating answers
# - retriever: The retriever (in this case, the FAISS vector store as a retriever) to fetch relevant documents
# - return_source_documents: Whether to include the source documents in the result
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)

# Define the first query
query = "What is the capital of France"
# Invoke the QA chain with the query
result = qa_chain.invoke(query)
# Print the answer obtained from the QA chain
print("Answer:", result["result"])
# Answer: Paris

# Define the second query
query = "Which is the longest river ?"
# Invoke the QA chain with the query
result = qa_chain.invoke(query)
# Print the answer obtained from the QA chain
print("Answer:", result["result"])
# Answer: The Nile