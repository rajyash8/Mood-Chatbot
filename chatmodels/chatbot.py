from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage

model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7
)

print("Choose Your AI Mode:")
print("1 for Angry Mode")
print("2 for Sad Mode")
print("3 for Funny Mode")

choice = int(input("Enter your choice: "))

if choice == 1:
    mode = "You are an angry AI assistant."
elif choice == 2:
    mode = "You are a sad AI assistant."
else :
    mode = "You are a funny AI assistant."


print("-------- Type 0 to exit the chat ---------")

messages = [
    SystemMessage(content=mode)

]

while True:
    prompt = input("You: ")

    if prompt == "0":
        break

    if not prompt.strip():
        continue

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)