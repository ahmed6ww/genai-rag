import os
import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Loading Github Repo
loader = GithubFileLoader(
    repo="panaversity/learn-applied-generative-ai-fundamentals",
    branch="main",
    access_token=GITHUB_PERSONAL_ACCESS_TOKEN,
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(".md"),
)
docs = loader.load()

# Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

# Create Embeddings and Vectorstore
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}
)
prompt = hub.pull("rlm/rag-prompt")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Define Streamlit Interface
st.title('Panaversity Generative AI Fundamentals')
st.write("Enter your query below:")

user_query = st.text_input("Query", "")

if user_query:
    # Processing Query
    docs = retriever.invoke(user_query)
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    context=formatted_docs
    # Construct the RAG Chain
    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain
    response = rag_chain.invoke({"context": formatted_docs, "question": user_query})
    st.write("Response:")
    st.write(response)