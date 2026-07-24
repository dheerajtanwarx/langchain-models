import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# project root ke andr env file ka path or parent[0] ka mtlb hota h current file ka folder or parent[1] ka mtlb hota h current file ke folder ka parent folder i.e Langchain-models in which we env 
# override true ka mtlb hai Agar terminal ya system me pehle se koi value set hai, jaise: AZURE_OPENAI_ENDPOINT=old_wrong_value. to .env wali latest value usko replace kar degi.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)


# print("endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print("api version:", os.getenv("AZURE_OPENAI_API_VERSION"))

model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

result = model.invoke("What is the capital of India?")

print(result.content)