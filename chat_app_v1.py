import streamlit as st
import time

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

st.title("Retrieval Chatbot")

# Chat history is stored in a session state to persist over reruns
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Layout: Chat display area and Input area
chat_container = st.container()
input_container = st.container()

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

