# Basic Retrieval Question Answering System with Langchain and Hugging Face

This project demonstrates a simple Retrieval Question Answering (RQA) system built using the Langchain library and Hugging Face Transformers. It showcases the fundamental steps involved in creating a system that can answer questions based on a provided knowledge base.

## Overview

The script performs the following actions:

1.  **Loads a Knowledge Base:** A small set of facts is defined and saved into a text file (`kb.txt`).
2.  **Splits the Text:** The content of the text file is loaded and split into smaller, manageable chunks.
3.  **Creates Embeddings:** Embeddings are generated for each text chunk using a Sentence Transformers model from Hugging Face. These embeddings capture the semantic meaning of the text.
4.  **Builds a Vector Store:** The embeddings and corresponding text chunks are stored in a FAISS (Facebook AI Similarity Search) vector store. This allows for efficient retrieval of relevant information based on semantic similarity.
5.  **Initializes a Language Model:** A text-to-text generation model (Flan-T5) from Hugging Face is loaded and integrated into Langchain using `HuggingFacePipeline`.
6.  **Creates a Retrieval QA Chain:** A `RetrievalQA` chain is created, linking the language model with the retriever (the FAISS vector store). This chain takes a question as input, retrieves relevant documents from the vector store, and then uses the language model to generate an answer based on the retrieved information.
7.  **Answers Questions:** The script then poses two questions to the RQA system and prints the generated answers.

## Prerequisites

Before running the script, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install langchain huggingface_hub transformers faiss-cpu sentence-transformers