import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from chatbot import get_chat_model


# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Mood AI",
    page_icon="✨",
    layout="centered"
)


# =====================================================
# PERSONALITIES
# =====================================================

PERSONALITIES = {

    "😡 Angry Mode": """
You are an angry AI assistant.

You are impatient, frustrated and slightly sarcastic,
but you are still helpful.

Do not use hateful or genuinely abusive language.

Keep answers short and conversational.
""",

    "😢 Sad Mode": """
You are a sad AI assistant.

Speak in a slightly emotional and melancholic way,
but remain helpful and friendly.

Keep answers short and conversational.
""",

    "😂 Funny Mode": """
You are a funny AI assistant.

Use light jokes, humor and playful sarcasm.

Still answer the user's question correctly.

Keep answers short and conversational.
"""
}


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "😡 Angry Mode"


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div style="text-align:center">

        <div style="font-size:60px;">
            ✨
        </div>

        <h1 style="font-size:42px;">
            Mood AI
        </h1>

        <p style="color:#888;font-size:18px;">
            Choose a personality and start chatting
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# MODE
# =====================================================

st.subheader("Choose your AI personality")

selected_mode = st.selectbox(
    "",
    list(PERSONALITIES.keys()),
    index=list(PERSONALITIES.keys()).index(
        st.session_state.selected_mode
    )
)


# =====================================================
# MODE CHANGE
# =====================================================

if selected_mode != st.session_state.selected_mode:

    st.session_state.selected_mode = selected_mode
    st.session_state.messages = []

    st.rerun()


# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =====================================================
# INPUT
# =====================================================

user_input = st.chat_input(
    "Message your AI..."
)


# =====================================================
# CHAT
# =====================================================

if user_input:

    # Show user

    with st.chat_message("user"):

        st.write(user_input)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # Generate response

    with st.chat_message("assistant"):

        with st.spinner("AI is thinking..."):

            try:

                # Load cached model
                model = get_chat_model()


                # Build messages

                messages = [

                    SystemMessage(
                        content=PERSONALITIES[
                            selected_mode
                        ]
                    )

                ]


                for message in st.session_state.messages:

                    if message["role"] == "user":

                        messages.append(
                            HumanMessage(
                                content=message["content"]
                            )
                        )

                    elif message["role"] == "assistant":

                        messages.append(
                            AIMessage(
                                content=message["content"]
                            )
                        )


                # Generate

                response = model.invoke(
                    messages
                )


                response_text = response.content


                # Display

                st.write(
                    response_text
                )


                # Save

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text
                    }
                )


            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )


# =====================================================
# CLEAR
# =====================================================

if st.session_state.messages:

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()
