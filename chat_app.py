import settings
import streamlit as st
from dotenv import load_dotenv
import json
import os
from docsrag.chromadb import read_db, get_retriever, list_collections
from docsrag.chain import get_chain, retrieve_docs, generate_response, generate_response_stream


def get_sources(docs):
    global base_url, repo_name
    sources = []
    docs_dir = f"{settings.APP_DATA}/{repo_name}"
    for doc in docs:
        base_path = doc.metadata.get("source")
        base_path = os.path.splitext(base_path)[0]
        common = os.path.commonpath([base_path, docs_dir])
        base_path = base_path.replace(common, "")
        if base_path[0] == "/":
            base_path = base_path[1:]
        if base_url[-1] == "/":
            base_url = base_url[:-1]
        url = os.path.join(base_url, base_path)
        if base_path.endswith("README"):
            url = url[:-7]
        sources.append(url)
    return sources


@st.cache_resource
def get_vectorstore(collection_name):
    import time
    t0 = time.time()
    vec = read_db(collection_name, settings.CHROMA_DB_DIR, settings.EMBEDDING_MODEL_NAME)
    print(f"Loaded vectorstore in {time.time()-t0:.2f} seconds.")
    n = vec._collection.count()
    print(f"The vectorstore has {n} vectors.")
    return vec


# Init
if "vectorstore" not in st.session_state:
    load_dotenv(".env")
    with open("openai_models.json") as f:
        models = json.load(f)
    st.session_state.models = models
    with open(settings.APP_DATA+"/collections.json") as f:
        collections = json.load(f)
    st.session_state.collections = collections
models = st.session_state.models
collections = st.session_state.collections

# Sidebar
collection_list = list_collections(settings.CHROMA_DB_DIR)
collection_name = st.sidebar.selectbox("Collection", map(lambda x: x.name, collection_list))
base_url = str(collections[collection_name]["base_url"])
repo_name = str(collections[collection_name]["repo_url"]).split("/")[-1]
st.sidebar.caption(base_url)
st.sidebar.caption(repo_name)
k = st.sidebar.number_input("K docs", value=4, step=1, min_value=1, max_value=10)
retrieve_only = st.sidebar.checkbox("Retrieve only", value=False)
if not retrieve_only:
    llm = st.sidebar.selectbox("LLM model", models.keys())
    desc = st.sidebar.caption(str(models[llm]))
    temperature = st.sidebar.slider("Temperature", value=0.7, step=0.01, min_value=0.0, max_value=1.0)

vectorstore = get_vectorstore(collection_name)
retriever = get_retriever(vectorstore, k=k)
if not retrieve_only:
    chain = get_chain(retriever, model=llm, temperature=temperature)


st.title("Superivsely SDK Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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

        response = ""
        if not retrieve_only:
            # Generate response
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