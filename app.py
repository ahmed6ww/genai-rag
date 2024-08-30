import streamlit as st
from langchain_community.document_loaders import GithubFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
# from langchain_community.retrievers import FAISS
# Define the GitHub repository details
repo = "panaversity/learn-applied-generative-ai-fundamentals"
branch = "main"
access_token = ""  # Replace with your actual GitHub personal access token
github_api_url = "https://api.github.com"

# Load documents from the GitHub repository
loader = GithubFileLoader(
    repo=repo,  # the repo name
    branch=branch,  # the branch name
    access_token=access_token,
    github_api_url=github_api_url,
    file_filter=lambda file_path: file_path.endswith(".md"),  # load all markdown files.
)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Initialize GPT4All embeddings
embeddings_model = GPT4AllEmbeddings()

# Precompute the embeddings for the document chunks
embeddings = [embeddings_model.embed(chunk.page_content) for chunk in splits]
print(embeddings)

# Create a vector store from the chunks
# vectorstore = FAISS.from_embeddings(embeddings=embeddings, texts=[chunk.page_content for chunk in splits])

# # Initialize the retriever
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# # Streamlit UI
# st.title('LangChain Document Chunks with GPT4All')
# st.write('Below are the text chunks generated from the documents in the GitHub repository.')

# for i, chunk in enumerate(splits):
#     st.subheader(f'Chunk {i+1}')
#     st.write(chunk.page_content)

# # Testing specific query
# query = "What are the approaches to Task Decomposition?"
# retrieved_docs = retriever.invoke(query)

# st.subheader('Search Results for the query: "What are the approaches to Task Decomposition?"')
# if retrieved_docs:
#     for i, doc in enumerate(retrieved_docs):
#         st.write(f"Result {i+1}: {doc.page_content}")
# else:
#     st.write("No results found.")

# # Display length of retrieved documents
# st.write(f"Number of retrieved documents: {len(retrieved_docs)}")

# # Print the content of the first retrieved document for verification
# if retrieved_docs:
#     st.write(f"Content of the first retrieved document: {retrieved_docs[0].page_content}")
