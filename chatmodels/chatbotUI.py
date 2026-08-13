import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Mood AI",
    page_icon="✦",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0b0f;
        color: #ffffff;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 850px;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 25px 0 20px 0;
    }

    .logo {
        width: 65px;
        height: 65px;
        margin: auto;
        border-radius: 20px;
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        box-shadow: 0 10px 40px rgba(124, 58, 237, 0.35);
    }

    .title {
        font-size: 38px;
        font-weight: 750;
        margin-top: 15px;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #888892;
        font-size: 15px;
        margin-top: 5px;
    }

    /* Mode cards */
    .mode-title {
        font-size: 14px;
        color: #a1a1aa;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* Chat area */
    [data-testid="stChatMessage"] {
        background: #15151b;
        border: 1px solid #24242d;
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 10px;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 15px;
    }

    /* Select box */
    div[data-baseweb="select"] > div {
        background-color: #15151b;
        border: 1px solid #2b2b35;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #2b2b35;
        background: #15151b;
        color: white;
    }

    .stButton button:hover {
        border-color: #7c3aed;
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #55555f;
        font-size: 12px;
        margin-top: 25px;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div class="logo">✦</div>
    <div class="title">Mood AI</div>
    <div class="subtitle">
        Choose a personality and start chatting
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# AI Modes
# -----------------------------
st.markdown(
    '<div class="mode-title">Choose your AI personality</div>',
    unsafe_allow_html=True
)

mode_options = {
    "😡 Angry Mode": "You are an angry AI assistant.",
    "😢 Sad Mode": "You are a sad AI assistant.",
    "😂 Funny Mode": "You are a funny AI assistant."
}


selected_mode = st.selectbox(
    "AI Mode",
    list(mode_options.keys()),
    label_visibility="collapsed"
)


# -----------------------------
# Initialize Session State
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = selected_mode

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content=mode_options[selected_mode]
        )
    ]


# -----------------------------
# Reset conversation when mode changes
# -----------------------------
if selected_mode != st.session_state.mode:

    st.session_state.mode = selected_mode

    st.session_state.messages = [
        SystemMessage(
            content=mode_options[selected_mode]
        )
    ]

    st.rerun()


# -----------------------------
# Model
# -----------------------------
model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7
)


# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)


# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input(
    "Message your AI..."
)


if prompt:

    # User message
    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = model.invoke(
                st.session_state.messages
            )

        st.markdown(response.content)

    # Save AI response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Powered by Mistral AI • Mood-based conversations
</div>
""", unsafe_allow_html=True)