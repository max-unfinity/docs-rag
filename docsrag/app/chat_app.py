import json
import os

import streamlit as st
from dotenv import load_dotenv

from docsrag.app.utils import upd_sqlite_version
from docsrag.rag.chain import (
    generate_response,
    generate_response_stream,
    get_chain,
    retrieve_docs,
)
from docsrag.rag.chromadb import get_retriever, read_db

# Uncomment if truobles with sqlite3 version
# More info: https://docs.trychroma.com/troubleshooting#sqlite"
upd_sqlite_version()


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


# Init
if "vectorstore" not in st.session_state:
    load_dotenv(".env")
    with open("config.json") as f:
        config = json.load(f)

    st.session_state.config = config
    st.session_state.vectorstore = read_db("supervisely-dev-portal-db")

    with open("models.json") as f:
        models = json.load(f)
    st.session_state.models = models
models = st.session_state.models

# Sidebar
k = st.sidebar.number_input("K docs", value=4, step=1, min_value=1, max_value=10)
llm = st.sidebar.selectbox("LLM model", models.keys())
desc = st.sidebar.caption(str(models[llm]))
temperature = st.sidebar.slider(
    "Temperature", value=0.7, step=0.01, min_value=0.0, max_value=1.0
)

retriever = get_retriever(st.session_state.vectorstore, k=k)
chain = get_chain(retriever, model=llm, temperature=temperature)


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
        docs = retrieve_docs(retriever, prompt)
        sources = get_sources(docs)
        references = "\n".join([f"- {source}" for source in sources])
        references = "**Retrieved docs:**\n" + references
        st.markdown(references)

        # response = generate_response(chain, prompt)
        # response = "**Answer:**\n\n" + response
        # st.markdown(response)

        # Simulate stream of response with milliseconds delay
        response = ""
        message_placeholder = st.empty()
        for chunk in generate_response_stream(chain, prompt):
            response += chunk
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)

    response = references + "\n\n" + response
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
