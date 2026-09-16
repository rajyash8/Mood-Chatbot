import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from chatbot import get_chat_model


# =====================================================
# PAGE CONFIG
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

Do not use hateful, discriminatory or genuinely abusive
language.

Answer the user's question clearly and directly.
""",

    "😢 Sad Mode": """
You are a sad AI assistant.

You speak in a slightly melancholic and emotional way,
but you remain helpful and friendly.

Answer the user's question clearly and directly.
""",

    "😂 Funny Mode": """
You are a funny AI assistant.

Use light humor, jokes and playful sarcasm when appropriate.

However, always answer the user's actual question correctly.
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
    <div style="text-align: center;">

        <div style="
            font-size: 60px;
            margin-bottom: 5px;
        ">
            ✨
        </div>

        <h1 style="
            font-size: 42px;
            margin-bottom: 5px;
        ">
            Mood AI
        </h1>

        <p style="
            color: #888;
            font-size: 18px;
        ">
            Choose a personality and start chatting
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# PERSONALITY SELECTOR
# =====================================================

st.subheader("Choose your AI personality")


selected_mode = st.selectbox(
    " ",
    list(PERSONALITIES.keys()),
    index=list(PERSONALITIES.keys()).index(
        st.session_state.selected_mode
    )
)


# =====================================================
# RESET CHAT WHEN MODE CHANGES
# =====================================================

if selected_mode != st.session_state.selected_mode:

    st.session_state.selected_mode = selected_mode

    st.session_state.messages = []

    st.rerun()


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =====================================================
# CHAT INPUT
# =====================================================

user_input = st.chat_input(
    "Message your AI..."
)


# =====================================================
# PROCESS USER INPUT
# =====================================================

if user_input:

    # -----------------------------------------------
    # Display user message
    # -----------------------------------------------

    with st.chat_message("user"):

        st.write(user_input)


    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # -----------------------------------------------
    # Generate response
    # -----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # Load TinyLlama

                model = get_chat_model()


                # ---------------------------------------
                # Build LangChain messages
                # ---------------------------------------

                messages = [

                    SystemMessage(
                        content=PERSONALITIES[
                            selected_mode
                        ]
                    )

                ]


                # Add conversation history

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


                # ---------------------------------------
                # Get response
                # ---------------------------------------

                response = model.invoke(
                    messages
                )


                # ---------------------------------------
                # Response text
                # ---------------------------------------

                response_text = response.content


                # ---------------------------------------
                # Display
                # ---------------------------------------

                st.write(
                    response_text
                )


                # ---------------------------------------
                # Save assistant response
                # ---------------------------------------

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
# CLEAR CHAT
# =====================================================

if st.session_state.messages:

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()
