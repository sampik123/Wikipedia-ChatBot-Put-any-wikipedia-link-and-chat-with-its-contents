from typing import List
import openai
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
import os
from urllib.parse import urlparse
from langchain.chat_models import ChatOpenAI
import uuid  # Import the uuid module

api = st.secrets['OPENAI_API_KEY']


# Function to retrieve text content from a Wikipedia link
def fetch_wikipedia_content(link: str) -> str:
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([p.get_text() for p in paragraphs])
    return content


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


# Function to clear chat history
def clear_chat_history():
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.", icon="🗑️")


# Set up the Streamlit app
st.title("MultiTurn ChatBot-Ask any questions from Wikipedia by providing the link")

# Sidebar for conversation history and new chat button
st.sidebar.title("Conversation History")

# Add a new chat button at the top of the sidebar
if st.sidebar.button("New Chat", key="new_chat_button"):
    clear_chat_history()

# Get or create a unique session ID for the current user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Use uuid.uuid4() to generate a unique identifier

# Initialize the chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Allow the user to input a Wikipedia link
wikipedia_link = st.sidebar.text_input("Enter Wikipedia Link to chat with its content:",
placeholder="Paste the Wikipedia link here")

if wikipedia_link and urlparse(wikipedia_link).scheme in ['http', 'https']:
    content = fetch_wikipedia_content(wikipedia_link)
    pages = text_to_docs(content)

    if pages:
        # Allow the user to select a page and view its content on the main page
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(label="Select Page", min_value=1, max_value=len(pages), step=1)
            st.write(pages[page_sel - 1])

        if api:
            # Test the embeddings and save the index in a vector database
            index = create_embeddings()

            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=api), chain_type="stuff",
                                             retriever=index.as_retriever())

            # Set up the conversational agent
            tools = [Tool(name="Wikipedia Q&A Tool", func=qa.run,
                          description="This tool allows you to ask questions about the Wikipedia article you've provided. You can inquire about various topics or information within the article.",
                          )]
            prefix = """Engage in a conversation with the AI, answering questions about the Wikipedia article. You have access to a single tool:"""
            suffix = """Begin the conversation!

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix,
                                                 input_variables=["input", "chat_history", "agent_scratchpad"])

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

            llm_chain = LLMChain(llm=ChatOpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"),
                                 prompt=prompt)

            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,
            memory=st.session_state.memory,
            handle_parsing_errors=True)

            # Allow the user to enter a query and generate a response
            query = st.text_input("Start a Conversation with the Bot!",
            placeholder="Ask the bot anything from Wikipedia")

            if query:
                # Update chat history
                role_user = "User"
                role_assistant = "Assistant"

                # Add user's message to the chat history
                st.session_state.chat_history.append({"role": role_user, "content": query})

                with st.spinner("Generating Answer to your Query: `{}`".format(query)):
                    # Generate response from the assistant
                    res = agent_chain.run(query)

                # Add assistant's response to the chat history
                st.session_state.chat_history.append({"role": role_assistant, "content": res})

                # Display the assistant's response
                st.info(res, icon="🤖")

# Display conversation history in the left sidebar
for message in st.session_state.chat_history:
    st.sidebar.text(f"{message['role']}: {message['content']}")
    

