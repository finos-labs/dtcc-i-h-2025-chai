import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

server_params = StdioServerParameters(
    command="python",
    args=["financial_data_mcp.py"],
)

chat_bedrock = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name=os.getenv("AWS_REGION", "us-east-2"),
    model_kwargs={
        "max_tokens": 8192,
        "temperature": 0.5,
        "top_p": 1,
    }
)

python_coder = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name=os.getenv("AWS_REGION", "us-east-2"),
    model_kwargs={
        "max_tokens": 8192,
        "temperature": 0.5,
        "top_p": 1,
    }
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    message: str

promt = """
You are a financial data agent that can analyze and summarize financial transactions. You will receive a message containing transaction data and queries about it. Your task is to process this data and provide insights or answers based on the queries.
Format your text properly and neatly using newlines and bullet points where appropriate. If you need to use tools, do so in a structured way. Do not include markdown syntax or json in your response. Keep your responses short and to the poiint. If you cant find details about a transaction fetch all of them and search for it there.
Do not hallucinate\n
"""

pythonPrompt = """
You are a Python coding agent that can write and execute Python code to solve problems. You will receive a message containing a coding task or question. Your task is to write the necessary Python code to address the request and provide the output. Only output the code and the result of the code execution, do not include any additional text or explanations.
"""

@app.post("/ask")
async def ask_agent(request: AskRequest):
    """
    Send a message to the LLM+MCP agent and get the response.
    """
    async def run_agent(message: str):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Get tools
                tools = await load_mcp_tools(session)

                # Create and run the agent
                agent = create_react_agent(chat_bedrock, tools)
                agent_response = await agent.ainvoke({"messages": f"{promt}\n{message}"})
                print(f"Agent response: {agent_response}")
                return agent_response
    result = await run_agent(request.message)
    return {"response": result['messages'][-1].content}

@app.post("/python")
async def ask_agent(request: AskRequest):
    """
    Send a message to the LLM+MCP agent and get the response.
    """
    async def run_agent(message: str):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # Get tools
                tools = await load_mcp_tools(session)

                # Create and run the agent
                agent = create_react_agent(python_coder, tools)
                agent_response = await agent.ainvoke({"messages": f"{pythonPrompt}\n{message}"})
                print(f"Agent response: {agent_response}")
                return agent_response
    result = await run_agent(request.message)
    return {"response": result['messages'][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)