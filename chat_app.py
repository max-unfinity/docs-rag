import streamlit as st
from src.chromadb import read_db, get_retriever
from src.chain import get_chain, run_chain
from dotenv import load_dotenv

k = st.sidebar.number_input("K retrieval", value=5, step=1, min_value=1, max_value=10)

# Init
if "vectorstore" not in st.session_state:
    load_dotenv(".env")
    st.session_state.vectorstore = read_db("supervisely-dev-portal-db")
    st.session_state.retriever = get_retriever(st.session_state.vectorstore, k=k)
    st.session_state.chain = get_chain(st.session_state.retriever)
    st.session_state.k = k

if "k" in st.session_state and st.session_state.k != k:
    st.session_state.retriever = get_retriever(st.session_state.vectorstore, k=k)
    st.session_state.chain = get_chain(st.session_state.retriever)
    st.session_state.k = k


st.title("Superivsely Dev Portal Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Type your message here...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # message_placeholder = st.empty()
        assistant_response, docs = run_chain(st.session_state.chain, st.session_state.retriever, prompt)
        for doc in docs:
            st.info(doc.metadata.get("source"))
            st.markdown(doc.page_content[:500])
            st.markdown("...")
        st.markdown(assistant_response)
        full_response = assistant_response
        # Simulate stream of response with milliseconds delay
        # for chunk in assistant_response.split():
        #     full_response += chunk + " "
        #     time.sleep(0.1)
        #     # Add a blinking cursor to simulate typing
        #     message_placeholder.markdown(full_response + "▌")
        # message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})