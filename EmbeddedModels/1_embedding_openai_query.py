from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=True)

embedding = AzureOpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)
result = embedding.embed_query("Delhi is the Capital of India")

print(str(result))