# 🧠 Gemini-powered Document Retriever

This project demonstrates how to build a **semantic search and Q&A system** using [Google Gemini (via LangChain)](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) with document embeddings and vector search.

It embeds historical content about the **Kennedy family**, creates a searchable vector database, and uses Gemini to answer questions based on retrieved context.

---

## 🚀 Features

- 🔐 Google Gemini (`gemini-2.0-flash`) via LangChain
- 📄 In-memory document collection
- 🔎 Semantic search using Google Embeddings
- 🧠 Question-answering with context-aware prompts
- 📦 Chroma vector store integration

---

## 📁 Project Structure

```bash
.
├── main.py                 # Main script with semantic search + Q&A chain
├── .env                    # Store your GOOGLE_API_KEY here
