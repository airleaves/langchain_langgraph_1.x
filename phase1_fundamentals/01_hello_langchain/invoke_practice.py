"""
invoke 方法深入实战 -配合README使用
=======================
本文件提供invoke方法的实战练习代码
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

#加载环境变量
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
)


def exercise_1_input_formats():
    """
    练习1：使用不同的输入格式调用模型

    """
    # 格式1：使用字符串输入
    response1 = model.invoke("你好！请用一句话介绍什么是人工智能。")
    print(response1.content)

    # 格式2：使用消息列表输入
    resopnse2=model.invoke([
        {
            "role": "system",
            "content": "你是一个智能助手"
        },
        {
            "role": "user",
            "content": "你好！请用一句话介绍什么是人工智能。"
        }
    ])
    print(resopnse2.content)

    from langchain_core.messages import SystemMessage,HumanMessage
    response3=model.invoke([
        SystemMessage(content="你是一个智能助手"),
        HumanMessage(content="你好！请用一句话介绍什么是人工智能。")
    ])
    print(response3.content)

def exercise_2_system_prompt():
    """
    练习2：使用系统提示设定AI行为
    通过不同的系统提示，让AI扮演不同的角色，观察回复的变化。

    """
    messages1=[
        {"role": "system", "content": "你是一个严肃的计算机教授，回答要学术化、专业化。"},
        {"role": "user", "content": "你好！请用一句话介绍什么是人工智能。"}
    ]
    response=model.invoke(messages1)
    print(response.content)

    question="太阳为什么东升西落？"
    messages2=[
        {"role": "system", "content": "你一个幽默的作者，回答要幽默化、诙谐化。"},
        {"role": "user", "content": question}
    ]

    response2=model.invoke(messages2)
    print(response2.content)

def exercise_3_conversation():
    """
    练习3：多轮对话
    练习目标 理解如何构建多轮对话
    关键:每次都要传递完整的对话历史
    """
    global_messages=[]
    messages=[
        {"role": "system", "content": "你一个幽默的作者，回答要幽默化、诙谐化。"},
        {"role": "user", "content": "你好！请用一句话介绍什么是人工智能。"}
    ]
    response=model.invoke(messages)
    print(response.content)
    global_messages.append(response.content)
    question2={"role": "user", "content": "人工智能的来源是什么？"}
    global_messages.append(question2)
    response2=model.invoke(global_messages)
    print(response2.content)
    global_messages.append(response2.content)
    question3={"role": "user", "content": "我的第一个问题是什么？"}
    global_messages.append(question3)
    response3=model.invoke(global_messages)
    print(response3.content)


def exercise_4_wrong_conversation():
    """
    练习目标： 理解多轮对话中常见错误
    演示：如果不传递历史，AI会"失忆"
    """
    # 错误示例 ：每次只传递最新问题，导致AI"失忆"
    respoonse1=model.invoke("我叫张三")
    print(respoonse1.content)
    response2=model.invoke("你叫什么名字？")
    print(response2.content)

    # 正确示例：每次传递完整的对话历史
    messages=[]
    response3=model.invoke("我叫张三")
    print(response3.content)
    messages.append(response3.content)
    messages.append({"role":"user","content":"我叫什么名字？"})
    response4=model.invoke(messages)
    print(response4.content)

def exercise_5_reponse_structure():
    """
    练习5：理解响应结构
    练习目标:理解invoke返回的AImessage对象
    学会访问各种有用的信息
    """
    response=model.invoke("你好！请用一句话介绍什么是人工智能。")
    #1主要信息
    print("主要信息:")
    print(f"reponse.content=",response.content)
    #2消息ID
    print("消息ID:")
    print(f"reponse.id=",response.id)
    #3响应源数据
    print("响应源数据:")
    metadata=response.response_metadata
    print(f"模型名称:",metadata.get("model_name"))
    print(f"结束原因:",metadata.get("finish_reason"))
    print(f"模型提供商:",metadata.get("model_provider"))

    #token使用情况
    print("token使用情况:")
    usage=metadata.get('token_usage',{})
    print(f"输入token数:",usage.get("prompt_tokens"))
    print(f"输出token数:",usage.get("completion_tokens"))
    print(f"总token数:",usage.get("total_tokens"))
    print(f"输入处理时间:",usage.get("prompt_time"))
    print(f"输出处理时间:",usage.get("completion_time"))

    # 5.计算成本
    print("计算成本:")
    total_tokens=usage.get("total_tokens",0)
    cost=total_tokens/1000*0.1
    print(f"本次调用成本为:",cost,"元")
    print("\n 提示:token统计对成本控制很重要")


def exercise_6_chatbot():
    """
    练习目标：综合运用所学知识，构建一个简单的聊天机器人
    """
    conversation=[
        {"role": "system", "content": "你一个幽默的作者，回答要幽默化、诙谐化。"},
    ]
    question=[
        "你是谁",
        "告诉我世界上最快的人是谁",
        "我叫张三你记住，最猛的男人",
        "我叫什么名字？",
    ]
    for q in question:
        conversation.append({"role": "user", "content": q})
        response=model.invoke(conversation)
        print(response.content)
        conversation.append({"role": "assistant", "content": response.content})

    print("结束",conversation)


def main():
    try:
        exercise_1_input_formats()
        input("\n 请按回车键继续...")
        exercise_2_system_prompt()
        input("\n 请按回车键继续...")
        exercise_3_conversation()
        input("\n 请按回车键继续...")
        exercise_4_wrong_conversation()
        input("\n 请按回车键继续...")
        exercise_5_reponse_structure()
        input("\n 请按回车键继续...")
        exercise_6_chatbot()

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()