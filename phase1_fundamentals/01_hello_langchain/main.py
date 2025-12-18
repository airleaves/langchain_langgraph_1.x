"""
Langchain 1.0 示例代码 - 第一个LLM调用示例
"""

import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import OpenAI,ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()# 加载环境变量
DEEPSEEK_API_KEY="sk-9d8eda40722f4910a8d5f4f8dd6f0e56"
KEY=os.getenv("DEEPSEEK_API_KEY")

model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY,
)

#示例1 最简单的LLM调用
def example_1_simple_invoke():
    """
    示例1：最简单的模型调用
    核心概念：
    - init_chat_model: 用于初始化聊天模型的统一接口
    - invoke: 同步调用模型的方法
    """
    response=model.invoke("你好！请用一句话介绍什么是人工智能。",config=None)
    print(response.content)


#示例2 使用消息列表进行对话
def example_2_messages():
    """
    示例2：使用消息列表
    核心概念：
    - SystemMessage: 系统消息，用于设定 AI 的行为和角色
    - HumanMessage: 用户消息
    - AIMessage: AI 的回复消息
    消息列表允许你构建多轮对话历史
    """
    messages=[
        SystemMessage(content="你是一个乐于助人的助手。"),
        HumanMessage(content="你好！请用一句话介绍什么是人工智能。"),    ]
    response=model.invoke(messages)
    print(response.content)
    messages.append(response)
    messages.append(HumanMessage(content="你能举一个人工智能的应用例子吗？"))
    response2=model.invoke(messages)
    print(response2.content)

#示例3 使用字典格式的参数
def example_3_dict_messages():
    """
    示例3：使用字典格式的参数
    核心概念：
    - 使用字典列表来表示消息
    - 每条消息包含 "role" 和 "content" 字段
    - langchain 1.0 支持更简洁的字典格式
    - 这种格式与OPENAI的API调用格式一致
    """
    messages=[
        {"role":"system","content":"你是一个乐于助人的助手。"},
        {"role":"user","content":"你好！请用一句话介绍什么是人工智能。"},
    ]
    response=model.invoke(messages)
    print(response.content)
    messages.append({"role":"assistant","content":response.content})
    messages.append({"role":"user","content":"你能举一个人工智能的应用例子吗？"})
    response2=model.invoke(messages)
    print(response2.content)

#示例4 配置模型参数
def example_4_model_parameters():
    """
    init_chat_model 支持的常用参数：
    - temperature: 控制输出的随机性（0.0-2.0）
      * 0.0: 最确定性，输出几乎不变
      * 1.0: 默认值，平衡创造性和一致性
      * 2.0: 最随机，最有创造性
    - max_tokens: 限制输出的最大 token 数量
    - model_kwargs: 传递给底层模型的额外参数
    """

    model=init_chat_model(
        model="deepseek:deepseek-chat",
        api_key=KEY,
        temperature=1,
        max_tokens=100,
        model_kwargs={"stop":["\n"]}
    )
    prompt="请用一句话介绍什么是人工智能。"
    for i in range(2):
        response=model.invoke(prompt)
        print(f"第{i+1}轮对话：",response.content)

if __name__ == "__main__":
    # example_1_simple_invoke()
    # example_2_messages()
    # example_3_dict_messages()
    example_4_model_parameters()