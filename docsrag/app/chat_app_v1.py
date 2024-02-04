import streamlit as st
import time
import hashlib

from pathlib import Path
from streamlit_tree_select import tree_select

from docsrag.rag.docs_reader import docs_tree, copy_files_to_folder_if_not_exists
from docsrag.app.utils import prepare_vectorstore, hash_from_str_list, upd_sqlite_version


# Uncomment if truobles with sqlite3 version
# More info: https://docs.trychroma.com/troubleshooting#sqlite"
upd_sqlite_version()

# Function to simulate chatbot response (to be replaced with actual retrieval logic)
def get_bot_response(user_input):
    time.sleep(1)  # Simulating processing time
    return f"Echoing '{user_input}'"

def send_message():
    user_input = st.session_state.input
    if user_input:
        # Update chat history with the user's message
        st.session_state.chat_history.append(f"{user_input}")

        # Get and display bot response
        bot_response = get_bot_response(user_input)
        st.session_state.chat_history.append(bot_response)

        # Clear the input box after sending the message
        # st.session_state.input = ""

        # Rerun the app to update the chat display
        # st.experimental_rerun()


def load_files_tree(url: str):
    name = url.split("/")[-1]
    path = f"./{name}"
    nodes = docs_tree(url, path)
    st.session_state.nodes = [{"label": name, "value": path, "children": nodes}]


st.session_state.nodes = []

st.title("Retrieval Chatbot")

# Chat history is stored in a session state to persist over reruns
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Layout: Chat display area and Input area
expandable_settings = st.expander("Settings")
chat_container = st.container()
input_container = st.container()


# prepare vectorstore
with expandable_settings:
    url_input = st.text_input("Enter docs URL", placeholder="https://github.com/tiangolo/fastapi")
    if url_input:
        load_files_tree(url_input)
        name = st.session_state.nodes[0]["label"]

    with st.form("select_data"):
        selected_data = tree_select(st.session_state.nodes)
        load_new_data_button = st.form_submit_button("Upload")

        if load_new_data_button:
            text_files = [p for p in selected_data['checked'] if p.endswith(".md")]
            cur_hash = hash_from_str_list(text_files)
            hash_name = f"{name}_{cur_hash}"
            raw_data_path = Path(".") / hash_name / "raw_data"
            # insert loading a bar somehow
            with st.spinner("Preparing files..."):
                copy_files_to_folder_if_not_exists(
                    src_files=text_files, dst=raw_data_path
                )
            with st.spinner("Preparing vectorstore..."):
                vectorstore = prepare_vectorstore(hash_name, raw_data_path)

# Chat display area
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.info(message, icon="👤")
        else:
            st.success(message, icon="🤖")


# Input area - with user input and send button
# with input_container:
user_input = st.chat_input(placeholder="Type your message here...", key="input", on_submit=send_message)
