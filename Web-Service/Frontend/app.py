import streamlit as st
import requests

# Load environment variables for FastAPI endpoint
FASTAPI_ENDPOINT = "http://chat-be-service:8000"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Title with emojis
st.title("Movie Bot ✨")
st.sidebar.image("images/logo.jpeg", use_column_width=True)
st.sidebar.write("Welcome!")

# Function to get conversation string
def get_conversation_string():
    conversation_string = ""
    for entry in st.session_state['chat_history']:
        conversation_string += "Human: " + entry['user'] + "\n"
        conversation_string += "Bot: " + entry['response'] + "\n"
    return conversation_string

# Chat input
user_input = st.text_input("Your message:", key="user_input")
# Send button
if st.button("Send"):
    if user_input:
        # Get conversation history
        conversation_history = get_conversation_string()

        # Prepare the request data
        data = {
            "query": user_input,
            "conversation_string": conversation_history
        }

        # Make API call
        try:
            response = requests.post(f"{FASTAPI_ENDPOINT}/chat", json=data)
            if response.status_code == 200:
                bot_response = response.json()['response']
                movies_desc = response.json().get('movies_desc', '')
                st.session_state['chat_history'].append({
                    'user': user_input,
                    'response': bot_response,
                    'movies_desc': movies_desc
                })
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")

        # Rerun the app to update the chat display
        st.experimental_rerun()

# Custom CSS for chat messages
st.markdown("""
    <style>
    .user-message {
        background-color: brown;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        text-align: right;
    }
    .bot-message {
        background-color: gray;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        text-align: left;
    }
    .movies-desc {
        font-size: 0.9em;
        color: white;
        background-color: green;
        margin: 5px 5px 5px 20px;
        text-align: left;
        padding: 10px;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .chat-container .user-message {
        align-self: flex-end;
    }
    </style>
""", unsafe_allow_html=True)

# Display chat messages
for entry in st.session_state['chat_history'][::-1]:
    st.markdown(f'<div class="chat-container"><div class="user-message">👤{entry["user"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-container"><div class="bot-message">🤖{entry["response"]}</div><br/>Movies Referring to:<br/><div class="movies-desc">{entry["movies_desc"]}</div></div>', unsafe_allow_html=True)
