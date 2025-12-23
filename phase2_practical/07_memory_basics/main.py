"""
Langchain 1.0 - Memory Basics (内存管理基础)
===========================================

本模块重点讲解:
1，InMemorySaver -LangGraph 提供的内存管理
2，checkpointer 参数  -为Agent 添加内存
3，thread_id - 会话管理
4，多轮对话状态保持
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

KEY=os.getenv("deepseek_api_key")

model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY,
)

#创建一个简单的工具
@tool
def get_user_info(user_id:str)->str:
    """获取用户信息"""
    users={
        "123":"张三，25岁，工程师",
        "456":"李四，30岁，设计师",
    }
    return users.get(user_id,"用户不存在")


# ============================================================================
# 示例1: 没有内存的 Agent模型调用
# ============================================================================
def example_1_no_memory():
    """
    示例1: 没有内存的Agent -不记得之前的对话

    关键: 每次调用都是独立的
    """
    print("\n" + "="*70)
    print("示例 1：没有内存的 Agent")
    print("="*70)

    # 创建没有checkpoint的Agent
    agent=create_agent(
        model=model,
        tools=[get_user_info],
        system_prompt="你是一个 helpful 的助手。"
    )
    print("\n第一轮对话：")

    response1=agent.invoke({
        "messages":[
            {"role": "user", "content": "我叫张三"}
        ]
    })
    print(f"Agent:{response1['messages'][-1].content}")

    print("\n第二轮对话：")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫什么？"}]
    })
    print(f"Agent: {response2['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 不记得第一轮对话")
    print("  - 每次 invoke 都是全新的开始")
    print("  - 需要手动传入历史消息才能记住")

# ============================================================================
# 示例2: 使用内存的 Agent模型调用
# ============================================================================
def example_2_with_memory():
    """
    示例2: 使用InMemorySaver 添加短期内存

    关键:
    1. checkpointer=InMemorySaver()
    2. config={"configurable":{"thread_id":"xxx"}}
    """
    print("\n" + "=" * 70)
    print("示例 2：使用 InMemorySaver 添加内存")
    print("=" * 70)

    # 创建有内存的Agent
    agent=create_agent(
        model=model,
        tools=[get_user_info],
        checkpointer=InMemorySaver(),
        system_prompt="你是一个 helpful 的助手。",
    )
    # config中指定thread_id
    config={"configurable":{"thread_id":"conversation_1"}}

    print("\n第一轮对话：")
    response1=agent.invoke({
        "messages":[
            {"role": "user", "content": "我叫张三"}
        ]
    },config=config)
    print(f"Agent:{response1['messages'][-1].content}")
    print("\n第二轮对话：")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫什么？"}]
    },config=config)
    print(f"Agent: {response2['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 记录了第一轮对话")
    print("  - checkpointer 自动保存对话历史")
    print("  - thread_id 用于区分不同的会话")

# ============================================================================
# 示例3: 多轮对话状态保持(不同thread_id)
# ============================================================================
def example_3_multiple_threads():
    """
    示例3: 管理多个独立的会话

    关键: 不同的thread_id=不同的对话
    """
    print("\n" + "=" * 70)
    print("示例 3：多个独立会话")
    print("=" * 70)

    agent=create_agent(
        model=model,
        tools=[get_user_info],
        checkpointer=InMemorySaver(),
        system_prompt="你是一个 helpful 的助手。",
    )

    #会话1
    config1={"configurable":{"thread_id":"conversation_1"}}
    print("\n会话1：")
    response1=agent.invoke({
        "messages":[
            {"role": "user", "content": "我叫张三"}
        ]
    },config=config1)
    print(f"Agent:{response1['messages'][-1].content}")

    #会话2
    config2={"configurable":{"thread_id":"conversation_2"}}
    print("\n会话2：")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫alice"}]
    },config=config2)
    print(f"Agent: {response2['messages'][-1].content}")

    #回到会话1
    print("\n回到会话1：")
    response3 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫什么？"}]
    },config=config1)
    print(f"Agent: {response3['messages'][-1].content}")

    #回到会话2
    print("\n回到会话2：")
    response4 = agent.invoke({
        "messages": [{"role": "user", "content": "我叫什么？"}]
    },config=config2)
    print(f"Agent: {response4['messages'][-1].content}")
    print("\n关键点：")
    print("  - 不同 thread_id 的会话完全独立")
    print("  - Agent 能正确记住每个会话的内容")
    print("  - 适合多用户聊天场景")

# ============================================================================
# 示例4: 带工具的内存Agent
# ============================================================================
def example_4_with_tools():
    """
    示例4: 内存 + 工具调用
    Agent 能记住之前调用工具的结果
    """
    print("\n" + "=" * 70)
    print("示例 4：内存 + 工具调用")
    print("=" * 70)
    agent=create_agent(
        model=model,
        tools=[get_user_info],
        checkpointer=InMemorySaver(),
        system_prompt="你是一个 helpful 的助手。",
    )
    print("\n第一轮对话：")
    config={"configurable":{"thread_id":"conversation_1"}}
    response1=agent.invoke({
        "messages":[
            {"role": "user", "content": "查询用户123的信息"}
        ]
    },config=config)
    print(f"Agent:{response1['messages'][-1].content}")
    print("\n第二轮对话：")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "刚才查询的用户多大？"}]
    },config=config)
    print(f"Agent: {response2['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 记住了工具调用的结果")
    print("  - 不需要重新调用工具")
    print("  - 对话上下文包含工具使用历史")

# ============================================================================
# 示例5: 查看内存状态
# ============================================================================
def example_5_check_memory():
    """
    查看和理解内存中保存的内容

    理解 checkpointer 保存了什么
    """
    print("\n" + "=" * 70)
    print("示例 5：查看内存状态")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "inspect_thread"}}

    # 进行几轮对话
    print("\n进行对话...")
    agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        config=config
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢编程"}]},
        config=config
    )

    # 再次调用，查看返回的完整状态
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么？"}]},
        config=config
    )
    print("\n对话历史中的消息数量:", len(response['messages']))
    print("\n最近的消息：")
    for msg in response['messages'][-3:]:
        msg_type=msg.__class__.__name__
        content=msg.content[:50]+"..." if len(msg.content)>50 else msg.content
        print(f"  [{msg_type}] {content}")


    print("\n关键点：")
    print("  - checkpointer 保存完整的消息历史")
    print("  - response['messages'] 包含所有历史消息")
    print("  - 每次调用都会追加新消息")
# ============================================================================
# 示例6: 实际应用场景
# ============================================================================
def example_6_practical_use():
    """
    示例6: 实际应用场景 -客服机器人
    模拟一个记住用户信息的客服场景
    """
    print("\n" + "=" * 70)
    print("示例 6：实际应用 - 客服机器人")
    print("=" * 70)

    agent=create_agent(
        model=model,
        tools=[get_user_info],
        system_prompt="你是一个 helpful 的客服机器人。",
        checkpointer=InMemorySaver()
    )
    # config
    config={"configurable":{"thread_id":"customer_1"}}

    conversations=[
        "你好",
        "帮我查一下用户为123的信息",
        "这个用户多大",
        "联系方式是什么",
        "我刚才问的用户信息是谁"
    ]
    for i,conversation in enumerate(conversations,1):
        print(f"\n对话 {i}:")
        response=agent.invoke({
            "messages":[
                {"role": "user", "content": conversation}
            ]
        },config=config)
        print(f"Agent:{response['messages'][-1].content}")
        print("\n关键点：")
        print("  - Agent 记住了用户的 ID")
        print("  - Agent 记住了查询的结果")
        print("  - 实现了流畅的多轮对话")

# 主函数
def main():
    print("欢迎来到 LangChain 示例程序！")
    try:
        example_1_no_memory()
        input("\n按 Enter 继续...")
        example_2_with_memory()
        input("\n按 Enter 继续...")
        example_3_multiple_threads()
        input("\n按 Enter 继续...")
        example_4_with_tools()
        input("\n按 Enter 继续...")
        example_5_check_memory()
        input("\n按 Enter 继续...")
        example_6_practical_use()

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

