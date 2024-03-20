import settings
import streamlit as st
from dotenv import load_dotenv
import json
import os
from docsrag.chromadb import read_db, list_collections
from docsrag.chain import get_chain, retrieve_docs, generate_response_stream


def get_sources(docs, base_url, repo_name):
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
    collections = get_collections()
    hf_model_name = collections[collection_name]["embedding_model"]
    import time
    t0 = time.time()
    vec = read_db(collection_name, settings.CHROMA_DB_DIR, hf_model_name)
    print(f"Loaded vectorstore in {time.time()-t0:.2f} seconds.")
    n = vec._collection.count()
    print(f"The vectorstore has {n} vectors.")
    return vec


@st.cache_data
def get_openai_models():
    with open("openai_models.json") as f:
        models = json.load(f)
    return models


@st.cache_data
def get_collections():
    with open(settings.APP_DATA+"/collections.json") as f:
        collections = json.load(f)
    return collections


# Initialize
load_dotenv(".env")
models = get_openai_models()
collections = get_collections()
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    collection_list = list_collections(settings.CHROMA_DB_DIR)
    collection_name = st.selectbox("Collection", map(lambda x: x.name, collection_list))
    used_collection = collections[collection_name]
    repo_name = used_collection["repo_url"].split("/")[-1]
    base_url = used_collection["base_url"]
    st.caption(f'Repository: "{repo_name}"')
    st.caption(f"Embedding model: {collections[collection_name]['embedding_model']}")
    if base_url:
        st.caption(f"Docs website url: {base_url}")
    k = st.number_input("Chunks to retrieve", value=4, step=1, min_value=1, max_value=10)
    retrieve_only = st.checkbox("Retrieve only", value=False)
    st.divider()
    if not retrieve_only:
        llm = st.selectbox("LLM model", models.keys())
        with st.expander("Model details"):
            st.code(json.dumps(models[llm], indent=0)[1:-1], language="python")
        temperature = st.slider("Temperature", value=0.7, step=0.01, min_value=0.0, max_value=1.0)


# Main content
st.title("Docs RAG")
st.caption("A chatbot that uses a Retrieval-Augmented Generation to answer questions based on documentation.")
vectorstore = get_vectorstore(collection_name)
retriever = vectorstore.as_retriever(search_kwargs={"k": k})
if not retrieve_only:
    chain = get_chain(retriever, model=llm, temperature=temperature)

prompt = st.chat_input("Type your message here...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        docs = retrieve_docs(retriever, prompt)
        # sources = get_sources(docs, base_url, repo_name)
        sources = [doc.metadata.get("source") for doc in docs]
        for doc, source in zip(docs, sources):
            md = doc.page_content
            with st.expander(source):
                st.markdown(md)
        # references = "\n".join([f"- {source}" for source in sources])
        # references = "**Retrieved docs:**\n" + references
        # st.markdown(references)

        response = ""
        if not retrieve_only:
            # Generate response
            message_placeholder = st.empty()
            for chunk in generate_response_stream(chain, prompt):
                response += chunk
                # blinking cursor to simulate typing
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(response)
    
    # response = references + "\n\n" + response
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.messages.append({"role": "assistant", "content": response})