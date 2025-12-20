"""
LangChain 1.0 - 消息类型与对话管理
================================

本模块重点讲解
1. 三种消息类型的实际使用
2. 对话历史管理(核心难点)
3.消息的修剪和优化

"""

import os
from  dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage
from langchain.chat_models import init_chat_model\

load_dotenv()

KEY=os.getenv("DEEPSEEK_API_KEY")

model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY
)
# ====================================================
# 示例1 三种消息类型
# ====================================================

def example_1_message_types():
    """
    三种消息类型: SystemMessage ,HumanMessage,AIMessage
    重点: 字典格式 VS 消息对象(推荐使用字典)
    """
    #方法1 消息对象
    messages_obj=[
        SystemMessage(content="你是一个Python 大师"),
        HumanMessage(content="python中的列表和java中的列表区别")
    ]
    response=model.invoke(messages_obj)
    print(f"AI回复:{response.content[:100]}...\n")

    #方法2 字典格式(推荐,简洁)
    messages_dict=[
        {"role":"system","content":"你是一个Python 大师"},
        {"role":"user","content":"python中的列表和java中的列表区别"}
    ]
    response=model.invoke(messages_dict)
    print(f"AI回复:{response.content[:100]}...\n")

    print("\n 推荐:直接用字典，更简洁")

# ====================================================
# 示例2 对话历史管理(核心难点)
# ====================================================
def example_2_conversation_history():
    """
    难点: 如何正确管理对话历史

    关键: 每次调用都要传递完整历史！
    """
    conversation=[
        {"role":"system","content":"你是一个简洁的助手，回答限制在50字以内"}
    ]
    # 第一次调用
    conversation.append({"role":"user","content":"什么是机器学习？"})
    response=model.invoke(conversation)
    print(f"AI回复:{response.content[:100]}...\n")

    #关键: 保存AI 回复到历史
    conversation.append({"role":"assistant","content":response.content})

    # 第2次调用
    conversation.append({"role":"user","content":"它有什么特点？"})
    response=model.invoke(conversation)
    print(f"AI回复:{response.content[:100]}...\n")

    conversation.append({"role":"assistant","content":response.content})

    # 第3次调用
    conversation.append({"role":"user","content":"我第一个问题是什么？"})
    response=model.invoke(conversation)
    print(f"AI回复:{response.content[:100]}...\n")

    print(f"\n 对话历史共{len(conversation)}条信息")
    print("AI 记住了之前的内容，因为每次都传递了完整历史")


# ============================================================================
# 示例 3：错误示范 - AI 失忆
# ============================================================================
def example_3_wrong_way():
    """
    错误示范：不保存对话历史

    结果：AI 会"失忆"
    """
    print("\n" + "="*70)
    print("示例 3：错误示范 - AI 失忆")
    print("="*70)

    print("\n❌ 错误做法：不保存历史")

    # 第一次
    r1 = model.invoke("我叫张三")
    print(f"用户: 我叫张三")
    print(f"AI: {r1.content[:50]}...")

    # 第二次（没有传递历史）
    r2 = model.invoke("我叫什么名字？")
    print(f"\n用户: 我叫什么名字？")
    print(f"AI: {r2.content[:80]}...")
    print("\n❌ AI 不记得你叫张三！")


# ============================================================================
# 示例 4 对话历史的优化
# ============================================================================
def example_4_conversation_history():
    """
    难点：对话历史太长怎么办？

    解决方案:
        1，只保留最近N条
        2. 总是保留system消息

    """
    def keep_recent_messages(messages,max_pairs=3):
        """
        保留最近的N轮对话
        参数：
            03_messages: 完整消息列表
            max_pairs: 保留的对话轮数

        返回:
            优化后的消息列表
        """

        # 分离system消息和对话消息
        system_msgs=[m for m in messages if m.get("role")=="system"]
        conversation_msgs=[m for m in messages if m.get("role") !="system"]

        # 只保留最近的消息(每轮=user+assistant)

        max_messages=max_pairs*2

        recent_msgs=conversation_msgs[-max_messages:]

        # 重新组合 返回 system+最近对话
        return system_msgs+recent_msgs

    # 模拟长对话
    long_conversation=[
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "第1个问题"},
        {"role": "assistant", "content": "第1个回答"},
        {"role": "user", "content": "第2个问题"},
        {"role": "assistant", "content": "第2个回答"},
        {"role": "user", "content": "第3个问题"},
        {"role": "assistant", "content": "第3个回答"},
        {"role": "user", "content": "第4个问题"},
        {"role": "assistant", "content": "第4个回答"},
        {"role": "user", "content": "第5个问题"},
    ]
    print(f"原始消息数:{len(long_conversation)}")

    #优化 只保留最近两轮
    short_conversation=keep_recent_messages(long_conversation,max_pairs=2)
    print(f"优化后消息数:{len(short_conversation)}")

    #使用优化后的历史
    response=model.invoke(short_conversation)
    print(f"AI 回复:{response.content[:100]}...")

    print("\n 技巧: 对话太长时，只保留最近的几轮即可 ")

# ============================================================================
# 示例 5 实战 - 简单聊天机器人
# =============================================================================
def example_5_simple_chatbot():
    """
    实战：简单聊天机器人 - 能记住对话的聊天机器人
    """

    conversation=[
        {"role": "system", "content": "你是一个简单问答助手"}
    ]
    questions=[
        "我叫李明，今年25岁",
        "我喜欢编程",
        "我叫什么名字？",
        "我今年多大？",
        "我喜欢什么？"
    ]

    for i,q in enumerate(questions,1):
        print(f"\n ---第{i}轮对话---")
        print(f"用户: {q}")
        conversation.append({"role": "user", "content": q})

        response=model.invoke(conversation)
        print(f"AI: {response.content}...")
        conversation.append({"role": "assistant", "content": response.content})

    print(f"\n 总共{len(conversation)}条信息")
    print("AI 完美记住了所有信息")

# =============================================================================
# 主程序
# =============================================================================

def main():
    print("\n"+"="*70)
    print("LangChain 1.0 - 消息类型与对话管理")
    print("="*70)

    try:
        example_1_message_types()
        input("\n 请按回车键继续...")
        example_2_conversation_history()
        input("\n 请按回车键继续...")
        example_3_wrong_way()
        input("\n 请按回车键继续...")
        example_4_conversation_history()
        input("\n 请按回车键继续...")
        example_5_simple_chatbot()
    except Exception as e:
        print(f"发生错误:{e}")


if __name__ == "__main__":
    main()

