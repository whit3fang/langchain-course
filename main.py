from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """
        Schema for source used by agent
    """
    url:str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with source url"""
    answer:str = Field(description="The agents answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")

llm = ChatOpenAI(model='gpt-5')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print('Hello there!')
    result =  agent.invoke({
        'messages': [HumanMessage(content='Search for 3 job postings using langchain in Bangalore on linkedin and list thier detials here')]
    })
    print(result)

if __name__ == '__main__':
    main()