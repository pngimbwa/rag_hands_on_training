# Objective
To demonstrate how Retrieval-Augmented Generation (RAG) can enhance LLM performance by integrating external documents into a chatbot built with Python, ChromaDB, LangChain, OpenAI, and Gradio.

This exercise will guide you through the process of building, running, and interacting with your own RAG-enabled chatbot. The chatbot uses the GPT-4o model from OpenAI to generate responses based on both general knowledge and document-specific data.

# Introduction to the Tools

Before diving into the hands-on exercise, here’s a brief introduction to the key technologies used:

Python: A widely used programming language known for its simplicity and effectiveness in data science and machine learning applications.

ChromaDB: A vector database that stores and retrieves embeddings, making it efficient for handling document-based queries.

LangChain: A framework that helps manage interactions between language models and external data sources, such as document retrieval and processing.

OpenAI: The provider of the GPT-4o model, which powers the chatbot's ability to generate human-like responses.

Gradio: A Python-based UI library that allows users to interact with machine learning models through an easy-to-use web interface.

# Step 1: Prerequisites

Before starting, ensure you have installed all the software in the RAG/requirements.txt () file:
## Installation Steps:
```bash
# Clone the repository
git clone https://github.com/pngimbwa/rag_hands_on_training.git

# Navigate to the RAG directory
cd rag_hands_on_training/RAG

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
