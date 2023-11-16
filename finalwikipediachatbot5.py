import re
import time
from io import BytesIO
from typing import List
import openai
import requests
from bs4 import BeautifulSoup
import streamlit as st
import sqlite3
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from langchain.chat_models import ChatOpenAI
import uuid  # Import the uuid module

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
api = os.getenv("OPENAI_API_KEY")


# Function to retrieve text content from a Wikipedia link
def fetch_wikipedia_content(link: str) -> str:
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([p.get_text() for p in paragraphs])
    return content





# Function to initialize the database
def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    
    # Create the table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Function to add a message to the database
def add_to_db(session_id, role, content):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_history VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

# Function to retrieve chat history from the database for a specific session
def get_chat_history(session_id):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history WHERE session_id=?", (session_id,))
    history = [{"role": role, "content": content} for role, content in c.fetchall()]
    conn.close()
    return history


# Function to clear the conversation history for the current session
def clear_history(session_id):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()



# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources as metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks



# Define a function for the embeddings
@st.cache_data
def create_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✔")
    return index

# Set up the Streamlit app
st.title("MultiTurn ChatBot-Ask any questions from Wikipedia by providing the link")


# Initialize the database
init_db()

# Sidebar for conversation history and new chat button
st.sidebar.title("Conversation History")

# Get or create a unique session ID for the current user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Use uuid.uuid4() to generate a unique identifier

# Display conversation history in the left sidebar
history = get_chat_history(st.session_state.session_id)
for message in history:
    st.sidebar.text(f"{message['role']}: {message['content']}")

# Allow the user to input a Wikipedia link
wikipedia_link = st.sidebar.text_input("Enter Wikipedia Link:", placeholder="Paste the Wikipedia link here")

if wikipedia_link and urlparse(wikipedia_link).scheme in ['http', 'https']:
    content = fetch_wikipedia_content(wikipedia_link)
    pages = text_to_docs(content)

    if pages:
        # Allow the user to select a page and view its content
        page_sel = st.sidebar.number_input(label="Select Page", min_value=1, max_value=len(pages), step=1)
        st.sidebar.write(pages[page_sel - 1])

        if api:
            # Test the embeddings and save the index in a vector database
            index = create_embeddings()

            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=api), chain_type="stuff", retriever=index.as_retriever())

            # Set up the conversational agent
            tools = [Tool(name="Wikipedia Q&A Tool", func=qa.run,
                    description="This tool allows you to ask questions about the Wikipedia article you've provided. You can inquire about various topics or information within the article.",
                     )]
            prefix = """Engage in a conversation with the AI, answering questions about the Wikipedia article. You have access to a single tool:"""
            suffix = """Begin the conversation!

            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(tools,prefix=prefix, suffix=suffix,
                                                  input_variables=["input", "chat_history", "agent_scratchpad"])

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

            llm_chain = LLMChain(llm=ChatOpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"), prompt=prompt)

            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory,handle_parsing_errors=True)

            # Allow the user to enter a query and generate a response
            query = st.text_input("Start a Conversation with the Bot!", placeholder="Ask the bot anything from Wikipedia")

            if query:
                # Update chat history
                role_user = "User"
                role_assistant = "Assistant"
    
                # Add user's message to the chat history
                add_to_db(st.session_state.session_id, role_user, query)

                with st.spinner("Generating Answer to your Query: `{}`".format(query)):
                    # Generate response from the assistant
                    res = agent_chain.run(query)

                # Add assistant's response to the chat history
                add_to_db(st.session_state.session_id, role_assistant, res)

                # Display the assistant's response
                st.info(res, icon="🤖")

# Display conversation history in the left sidebar
history = get_chat_history(st.session_state.session_id)
for message in history:
    st.sidebar.text(f"{message['role']}: {message['content']}")


