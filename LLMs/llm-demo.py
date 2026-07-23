# pip install -qU langchain "langchain[openai]"
import os
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# llm = AzureChatOpenAI(
#     azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#     model=os.environ["AZURE_OPENAI_MODEL_NAME"],
# )


os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)