# import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())


# groq_api_key = os.environ["GROQ_API_KEY"]

# from langchain_groq import ChatGroq

# # Create LLM instance using Groq API
# llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# from langchain_core.documents import Document

# documents = [
#     Document(
#         page_content="John F. Kennedy served as the 35th president of the United States from 1961 until his assassination in 1963.",
#         metadata={"source": "us-presidents-doc"},
#     ),
#     Document(
#         page_content="Robert F. Kennedy was a key political figure and served as the U.S. Attorney General; he was also assassinated in 1968.",
#         metadata={"source": "us-politics-doc"},
#     ),
#     Document(
#         page_content="The Kennedy family is known for their significant influence in American politics and their extensive philanthropic efforts.",
#         metadata={"source": "kennedy-family-doc"},
#     ),
#     Document(
#         page_content="Edward M. Kennedy, often known as Ted Kennedy, was a U.S. Senator who played a major role in American legislation over several decades.",
#         metadata={"source": "us-senators-doc"},
#     ),
#     Document(
#         page_content="Jacqueline Kennedy Onassis, wife of John F. Kennedy, was an iconic First Lady known for her style, poise, and dedication to cultural and historical preservation.",
#         metadata={"source": "first-lady-doc"},
#     ),
# ]

# from langchain_chroma import Chroma

# # Import Hugging Face embeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# # Create Hugging Face embeddings
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore = Chroma.from_documents(documents,embedding=embedding_model)

# response = vectorstore.similarity_search("John")

# print("\n----------\n")

# print("Search for John in the vector database:")

# print("\n----------\n")
# print(response)

# print("\n----------\n")

# response = vectorstore.similarity_search_with_score("John")

# print("\n----------\n")

# print("Search for John in the vector database (with scores):")

# print("\n----------\n")
# print(response)

# print("\n----------\n")

# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 1},
# )

# response = retriever.batch(["John", "Robert"])

# print("\n----------\n")

# print("Search for John and Robert in the vector database (with vectorstore as retriever):")

# print("\n----------\n")
# print(response)

# print("\n----------\n")

# from typing import List

# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda

# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

# response = retriever.batch(["John", "Robert"])

# print("\n----------\n")

# print("Search for John and Robert in the vector database (select top result):")

# print("\n----------\n")
# print(response)

# print("\n----------\n")

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# message = """
# Answer this question using the provided context only.

# {question}

# Context:
# {context}
# """

# prompt = ChatPromptTemplate.from_messages([("human", message)])

# chain = {
#     "context": retriever, 
#     "question": RunnablePassthrough()} | prompt | llm

# response = chain.invoke("tell me about Jackie")

# print("\n----------\n")

# print("tell me about Jackie (simple retriever):")

# print("\n----------\n")
# print(response.content)

# print("\n----------\n")


# import os 
# from dotenv import load_dotenv, find_dotenv

# _ = load_dotenv(find_dotenv())

# groq_api_key = os.environ["GROQ_API_KEY"]

# from langchain_groq import ChatGroq

# llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# from langchain_core.documents import Document


# documents = [
#     Document(
#         page_content="John F. Kennedy served as the 35th president of the United States from 1961 until his assassination in 1963.",
#         metadata={"source": "us-presidents-doc"},
#     ),
#     Document(
#         page_content="Robert F. Kennedy was a key political figure and served as the U.S. Attorney General; he was also assassinated in 1968.",
#         metadata={"source": "us-politics-doc"},
#     ),
#     Document(
#         page_content="The Kennedy family is known for their significant influence in American politics and their extensive philanthropic efforts.",
#         metadata={"source": "kennedy-family-doc"},
#     ),
#     Document(
#         page_content="Edward M. Kennedy, often known as Ted Kennedy, was a U.S. Senator who played a major role in American legislation over several decades.",
#         metadata={"source": "us-senators-doc"},
#     ),
#     Document(
#         page_content="Jacqueline Kennedy Onassis, wife of John F. Kennedy, was an iconic First Lady known for her style, poise, and dedication to cultural and historical preservation.",
#         metadata={"source": "first-lady-doc"},
#     ),
# ]

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore = Chroma.from_documents(documents,embedding=embedding_model)

# response = vectorstore.similarity_search("John")

# print("\n----------\n")

# print("Search for John in the vector database:")

# print("\n----------\n")
# # print(response)

# print("\n----------\n")

# response = vectorstore.similarity_search_with_score("John")

# print("\n----------\n")

# print("Search for John in the vector database (with scores):")

# print("\n----------\n")
# # print(response)

# print("\n----------\n")


# from typing import List

# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda

# retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
# response = retriever.batch({"John", "Robert"})

# # print(response)

# retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":1})
# response = retriever.batch({"John", "Robert"})

# print(response)

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough

# message = """
# Answer this question using the provided context only.
# {question}
# Context:{context}
# """

# prompt = ChatPromptTemplate.from_messages([("human",message)])

# chain = {
#     "context":retriever,
#     "question": RunnablePassthrough()} | prompt | llm 


# response = chain.invoke("tell me about jackie")

# print(response.content)



import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# Replace GROQ with Google AI (Gemini) API key
google_api_key = os.environ["GOOGLE_API_KEY"]

# Import Gemini model from langchain
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.1
)

from langchain_core.documents import Document

# Same document collection
documents = [
    Document(
        page_content="John F. Kennedy served as the 35th president of the United States from 1961 until his assassination in 1963.",
        metadata={"source": "us-presidents-doc"},
    ),
    Document(
        page_content="Robert F. Kennedy was a key political figure and served as the U.S. Attorney General; he was also assassinated in 1968.",
        metadata={"source": "us-politics-doc"},
    ),
    Document(
        page_content="The Kennedy family is known for their significant influence in American politics and their extensive philanthropic efforts.",
        metadata={"source": "kennedy-family-doc"},
    ),
    Document(
        page_content="Edward M. Kennedy, often known as Ted Kennedy, was a U.S. Senator who played a major role in American legislation over several decades.",
        metadata={"source": "us-senators-doc"},
    ),
    Document(
        page_content="Jacqueline Kennedy Onassis, wife of John F. Kennedy, was an iconic First Lady known for her style, poise, and dedication to cultural and historical preservation.",
        metadata={"source": "first-lady-doc"},
    ),
]

# Use Google's text embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)

# Create vector store with Google embeddings
vectorstore = Chroma.from_documents(documents, embedding=embedding_model)

# Similarity search
response = vectorstore.similarity_search("John")
print("\n----------\n")
print("Search for John in the vector database:")
print("\n----------\n")
# print(response)

print("\n----------\n")
response = vectorstore.similarity_search_with_score("John")
print("\n----------\n")
print("Search for John in the vector database (with scores):")
print("\n----------\n")
# print(response)

print("\n----------\n")

from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
response = retriever.batch({"John", "Robert"})
# print(response)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
response = retriever.batch({"John", "Robert"})
print(response)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.
{question}

Context:{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

chain = {
    "context": retriever,
    "question": RunnablePassthrough()
} | prompt | llm

response = chain.invoke("tell me about jackie")
print(response.content)