# 1. Simple Text vs. Chat Messages (Roles)HuggingFaceEndpoint:
# Yeh ek raw text-completion model hai. Yeh nahi samajhta ki SystemMessage, HumanMessage, ya AIResponse kya hote hain. Yeh bas ek lamba string leta hai aur aage ka text predict kar deta hai.
# ChatHuggingFace: Yeh is raw model ke upar ek wrapper (layer) laga deta hai. Yeh aapke input ko Llama-3.1 ke specific chat format (jaise <|start_header_id|>user<|end_header_id|>) mein convert karta hai.


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm) #Yeh line basically aapke simple text model ko ek smart Chat Model mein badal deti hai jo Hugging Face ke standard patterns ko samajh sake.

result = model.invoke("What is Html?")

print(result.content)