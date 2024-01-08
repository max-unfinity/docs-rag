import streamlit as st
from dotenv import load_dotenv
import json
import os
from src.chromadb import read_db, get_retriever
from src.chain import get_chain, retrieve_docs, generate_response, generate_response_stream


def get_sources(docs):
    sources = []
    for doc in docs:
        base_url = st.session_state.config["base_url"]
        docs_dir = st.session_state.config["docs_dir"]
        base_path = doc.metadata.get("source")
        base_path = os.path.splitext(base_path)[0]
        base_path = base_path.replace(docs_dir, "")
        if base_path[0] == "/":
            base_path = base_path[1:]
        if base_url[-1] == "/":
            base_url = base_url[:-1]
        url = f"{base_url}/{base_path}"
        if base_path.endswith("README"):
            url = url[:-7]
        sources.append(url)
    return sources


k = st.sidebar.number_input("K retrieval", value=4, step=1, min_value=1, max_value=10)

# Init
if "vectorstore" not in st.session_state:
    load_dotenv(".env")
    with open("config.json") as f:
        config = json.load(f)
    st.session_state.config = config
    st.session_state.vectorstore = read_db("supervisely-dev-portal-db")
    st.session_state.retriever = get_retriever(st.session_state.vectorstore, k=k)
    st.session_state.chain = get_chain(st.session_state.retriever)
    st.session_state.k = k

if "k" in st.session_state and st.session_state.k != k:
    st.session_state.retriever = get_retriever(st.session_state.vectorstore, k=k)
    st.session_state.chain = get_chain(st.session_state.retriever)
    st.session_state.k = k


st.title("Superivsely SDK Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Type your message here...")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        docs = retrieve_docs(st.session_state.retriever, prompt)
        sources = get_sources(docs)
        references = "\n".join([f"- {source}" for source in sources])
        references = "**Retrieved docs:**\n" + references
        st.markdown(references)

        # response = generate_response(st.session_state.chain, prompt)
        # response = "**Answer:**\n\n" + response
        # st.markdown(response)

        # Simulate stream of response with milliseconds delay
        response = ""
        message_placeholder = st.empty()
        for chunk in generate_response_stream(st.session_state.chain, prompt):
            response += chunk
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)
    
    response = references + "\n\n" + response
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})