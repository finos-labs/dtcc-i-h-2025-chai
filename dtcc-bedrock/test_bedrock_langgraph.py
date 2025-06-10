import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

load_dotenv()

server_params = StdioServerParameters(
    command="python",
    args=["financial_data_mcp.py"],
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "List my 3 next reminders"})
            return agent_response

def test_bedrock_connection_langchain():
    """Test Bedrock connection using LangChain"""
    
    # Set AWS credentials as environment variables for LangChain
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    if os.getenv("AWS_SESSION_TOKEN"):
        os.environ["AWS_SESSION_TOKEN"] = os.getenv("AWS_SESSION_TOKEN")
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_REGION", "us-east-2")
    
    try:
        # Initialize ChatBedrock with LangChain
        chat_bedrock = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            model_kwargs={
                "max_tokens": 50,
                "temperature": 0.5,
                "top_p": 1,
            }
        )
        
        # Create message
        message = HumanMessage(content="Hello! What is the capital of France?")
        
        # Invoke the model
        response = chat_bedrock.invoke([message])
        
        print("Model response:")
        print(response.content)
        return True
        
    except Exception as e:
        print("Error invoking model:", e)
        return False

def test_bedrock_streaming():
    """Test streaming response from Bedrock using LangChain"""
    
    try:
        chat_bedrock = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            model_kwargs={
                "max_tokens": 100,
                "temperature": 0.5,
            }
        )
        
        message = HumanMessage(content="Tell me a short story about a robot.")
        
        print("Streaming response:")
        for chunk in chat_bedrock.stream([message]):
            print(chunk.content, end="", flush=True)
        print("\n")
        return True
        
    except Exception as e:
        print("Error with streaming:", e)
        return False

def test_with_chat_history():
    """Test conversation with chat history using LangChain"""
    
    try:
        chat_bedrock = ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            region_name=os.getenv("AWS_REGION", "us-east-2"),
            model_kwargs={
                "max_tokens": 100,
                "temperature": 0.3,
            }
        )
        
        # Conversation with multiple messages
        from langchain_core.messages import AIMessage
        
        messages = [
            HumanMessage(content="My name is John and I like pizza."),
            AIMessage(content="Nice to meet you, John! Pizza is delicious. What's your favorite type?"),
            HumanMessage(content="What's my name and what do I like?")
        ]
        
        response = chat_bedrock.invoke(messages)
        
        print("Chat history response:")
        print(response.content)
        return True
        
    except Exception as e:
        print("Error with chat history:", e)
        return False

if __name__ == "__main__":
    print("Testing basic LangChain Bedrock connection...")
    test_bedrock_connection_langchain()
    
    print("\n" + "="*50)
    print("Testing streaming...")
    test_bedrock_streaming()
    
    print("\n" + "="*50)
    print("Testing chat history...")
    test_with_chat_history()