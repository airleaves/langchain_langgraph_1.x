"""
Langchain 1.0 基础教程 - 提示词模板
=======================================
本文件演示如何使用Langchain的提示词模板系统
涵盖一下核心概念:
1.PromptTemplate -简单文本模板
2.ChatPromptTemplate - 聊天消息模板
3.模板变量和格式化
4.消息模板的组合
5.实际应用场景

作者:Langchain 学习者
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.prompts import (
SystemMessagePromptTemplate,
HumanMessagePromptTemplate
)

#加载环境变量
load_dotenv()
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

#初始化模型
model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
)

# ==========================================
# 示例1 为什么需要提示词？
# ==========================================

def example_1_why_templates():
    """
    示例1: 对比字符串拼接 VS 模板
    问题: 字符串拼接容易出错、难维护、不可复用
    解决: 使用提示词模板
    """
    topic="Python"
    difficulty="初学者"

    prompt_str=f"你是一个{difficulty}级别的编程专家，请用简单的语言介绍{topic}。"

    response=model.invoke(prompt_str)
    print(f"AI回复:{response.content[:100]}...\n")

    #推荐：使用PromptTemplate模板
    template=PromptTemplate.from_template(
        "你是一个A{difficulty}级别的编程专家，请用简单的语言介绍{topic}。"
    )

    prompt=template.format(difficulty=difficulty,topic=topic)

    response2=model.invoke(prompt)
    print(f"AI回复:{response2.content[:100]}...\n")

# ==========================================
# 示例2 PromptTemplate 基础用法
# ==========================================

def example_2_prompt_template_basic():
    """
    示例2 : PromptTemplate 的基本用法
    PromptTemplate 用于简单的文本模板
    适合单一提示词的场景
    """
    # 方法1: 使用from_template
    template1= PromptTemplate.from_template(
     "将以下文本翻译成{language}:\n{text}"
    )
    prompt=template1.format(language="English",text="你好，世界！")
    response1=model.invoke(prompt)
    print(f"AI回复:{response1.content[:100]}...\n")

    # 方法2: 显示指定变量(更严格）
    template2=PromptTemplate(
        input_variables=["product","feature"],
        template="请为{product}写一句广告语，重点突出{feature}的特点"
    )
    prompt=template2.format(product="手机",feature="拍照")
    response2=model.invoke(prompt)
    print(f"AI回复:{response2.content[:100]}...\n")

    # 方法3: invoke直接调用
    template3=PromptTemplate.from_template(
        "请为{product}写一句广告语，重点突出{feature}的特点"
    )
    prompt=template3.format(product="手机",feature="拍照")
    response3=model.invoke(prompt)
    print(f"AI回复:{response3.content[:100]}...\n")

# ==========================================
# 示例3 ChatPromptTemplate 聊天消息模板
# ==========================================
def example_3_chat_prompt_template():
    """
    示例3:ChatPromptTemplate 的基本用法
    ChatPromptTemplate 用于构建聊天消息
    支持 system、user、assistant 多种角色
    """
    chat_template=ChatPromptTemplate.from_messages([
        ("system","你是一个{role},擅长{expertise}"),
        ("user","请帮我{text}")
    ])
    print(f"模板变量:{chat_template.input_variables}")

    # 格式化模板
    messages=chat_template.format_messages(
        role="程序员",
        expertise="机器学习",
        text="请写一个机器学习代码"
    )
    print("\n生成的消息:")
    for msg in messages:
        print(f"{msg.type}:{msg.content}")

    response= model.invoke(messages)
    print(f"AI回复:{response.content[:150]}...\n")

# ==========================================
# 示例4 多轮对话模板
# ==========================================
def example_4_conversation_template():
    """
    示例4: 构建多轮对话的模板
    包含系统提示、对话历史和当前问题
    """
    template=ChatPromptTemplate.from_messages([
        ("system","你是一个{role},{instruction}"),
        ("user","{question1}"),
        ("assistant","{answer1}"),
        ("user","{question2}"),
    ])

    print("模板结构")
    print("1.System:设定角色和指令")
    print("2.User:第一个问题")
    print("3.Assistant:模型回复")
    print("4.User:第二个问题(基于上下文)")

    # 填充模板
    messages = template.format_messages(
        role="Python 专家",
        instruction="回答要简洁、准确",
        question1="什么是列表？",
        answer1="列表是 Python 中的有序可变集合，用方括号 [] 表示。",
        question2="它和元组有什么区别？"  # 基于上下文的问题
    )
    print("\n生成的消息:")
    for i, msg in enumerate(messages,1):
        content_preview =msg.content[:60]+"..." if len(msg.content) > 60 else msg.content
        print(f"{i} {msg.type}:{content_preview}")
    reponse=model.invoke(messages)
    print(f"AI回复:{reponse.content}\n")

# ==========================================
# 示例5 使用MessagePromptTemplate(高级)
# ==========================================
def example_5_message_template():
    """
    示例5 使用MessagePromptTemplate 类
    提供更细粒度的控制
    """

    # 分别创建不同类型的消息模板
    system_template=SystemMessagePromptTemplate.from_template(
        "你是一个{role},擅长{expertise}"
    )
    human_template=HumanMessagePromptTemplate.from_template(
        "关于{topic}，我想知道{question}"
    )

    # 组合成ChatPromptTemplate
    chat_template=ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])

    print("模板组件:")
    print(f"\n总变量:{chat_template.input_variables}\n")

    # 使用模板
    messages=chat_template.format_messages(
        role="Python 专家",
        expertise="机器学习",
        topic="机器学习",
        question="什么是列表？"
    )
    response=model.invoke(messages)
    print(f"AI回复:{response.content}\n")

# ===========================================
# 示例6 部分变量(Partial Variables)
# ===========================================
def example_6_partial_variables():
    """
    示例6 部分变量(Partial Variables)
    适用场景：
    - 某些变量固定不变
    - 需要创建模板变体
    """

    # 创建原始模板
    original_template=ChatPromptTemplate.from_messages([
        ("system","你是一个{role},你的目标用户是{audience}"),
        ("user","请{task}")
    ])

    print(f"原始模板变量:{original_template.input_variables}\n")


    #部分填充，固定role和audience
    partially_filled=original_template.partial(
        role="科技博客作者",
        audience="程序员"
    )
    print(f"部分填充后的变量:{partially_filled.input_variables}\n")

    # 提供task
    message1=partially_filled.format_messages(
        task="写一篇关于机器学习的博客"
    )
    response1=model.invoke(message1)
    print(f"文章1回复:{response1.content}\n")

    # 复用模板 ，不同的task
    messages2=partially_filled.format_messages(
        task="写一篇关于深度学习的博客"
    )
    response2=model.invoke(messages2)
    print(f"文章2回复:{response2.content}\n")

# ===========================================
# 示例7 与LCEL 链式调用(预览)
# ===========================================
def example_7_lcel_chains():
    """
    示例7 与LCEL 链式调用(预览)
    Langchain Expression language(LCEL)
    """
    template=ChatPromptTemplate.from_messages([
        ("system","你是一个{role}"),
        ("user","{input}")
    ])

    # 使用|运算符创建链
    chain= template|model

    response=chain.invoke({
        "role":"Python 专家",
        "input":"请写一个机器学习代码"
    })
    print(f"AI回复:{response.content}\n")
    print("💡 链式调用的优势：")
    print("  1. 代码更简洁")
    print("  2. 组件可复用")
    print("  3. 易于调试和监控")


# ===========================================
# 主程序
# ===========================================
def main():
    try:
        example_1_why_templates()
        input("\n 请按回车键继续...")
        example_2_prompt_template_basic()
        input("\n 请按回车键继续...")
        example_3_chat_prompt_template()
        input("\n 请按回车键继续...")
        example_4_conversation_template()
        input("\n 请按回车键继续...")
        example_5_message_template()
        input("\n 请按回车键继续...")
        example_6_partial_variables()
        input("\n 请按回车键继续...")
        example_7_lcel_chains()
        print("\n 恭喜，所有示例已运行完毕！")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()