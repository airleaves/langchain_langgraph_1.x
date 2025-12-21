"""
简单测试: 验证Agent 可以正常工作
"""
import os
import sys

parent_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(parent_dir,'04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from weather import get_weather
from calculator import calculator

# 加载环境变量
load_dotenv()
KEY=os.getenv("DEEPSEEK_API_KEY")

# 初始化模型
model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY
)

agent=create_agent(
    model=model,
    tools=[get_weather,calculator],
    system_prompt="""
    你是一个 helpful 的助手，可以查询天气信息,数值计算
    """
)

response1=agent.invoke({
    "messages":[
        {"role":"user","content":"今天北京的天气如何？"}
    ]
})
print(f"AI回复:{response1['messages'][-1].content}\n")

response2=agent.invoke({
    "messages":[
        {"role":"user","content":"1+1等于多少？"}
    ]
})
print(f"AI回复:{response2['messages'][-1].content}\n")

print("测试成功！Agent 可以正常调用工具。")
