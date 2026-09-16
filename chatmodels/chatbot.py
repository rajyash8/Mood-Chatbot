from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline
)


def get_chat_model():

    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "do_sample": False,
            "repetition_penalty": 1.03,
        },
    )

    chat_model = ChatHuggingFace(
        llm=llm
    )

    return chat_model
