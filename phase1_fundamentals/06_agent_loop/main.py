"""
LangChain 1.0 - Agent 执行循环（ReAct 模式）
============================================

本模块重点讲解：
1. ReAct 执行循环的详细过程（Reason → Act → Observe）
2. 流式输出（streaming）
3. 查看中间步骤
4. 理解消息流转

⚠️ API 说明：
- 本模块使用 `create_react_agent`（来自 langgraph.prebuilt）
- 这是 LangGraph 1.0 的预构建图，明确使用 ReAct 架构
- 适合学习 Agent 执行循环的底层原理

🔄 ReAct 循环：
   Reason（推理）→ Act（行动）→ Observe（观察）→ 循环直到完成
"""
import os
import sys

# 添加工具目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # ✅ LangGraph 预构建 ReAct Agent
from calculator import calculator
from weather import get_weather

# 加载环境变量
load_dotenv()
KEY = os.getenv("DEEPSEEK_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 GROQ_API_KEY\n"
        "访问 https://console.groq.com/keys 获取免费密钥"
    )

# 初始化模型
model = init_chat_model("groq:llama-3.3-70b-versatile", api_key=KEY)

# ========================================================================
# 示例1:理解执行循环 - 查看完整消息历史
# ========================================================================
def example_1_understand_loop():
    """
    示例1: 查看Agent 执行循环的每一步

    关键；response['messages']包含完整的对话历史
    """
    print("\n" + "=" * 70)
    print("示例 1：Agent 执行循环详解")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="你是一个有帮助的助手。"
    )
    print("\n问题：25 乘以 8 等于多少？")
    response=agent.invoke({
        "messages":[
            {"role": "user", "content": "25 乘以 8 等于多少？"}
        ]
    })

    print("\n完整消息历史：")
    for i,message in enumerate(response['messages'],1):
        print(f"\n{'=' * 60}")
        print(f"消息 {i}: {msg.__class__.__name__}")
        print(f"{'=' * 60}")

        if hasattr(message, 'content') and message.content:
            print(f"内容：{message.content}")

        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"工具调用：{message.tool_calls}")

        if hasattr(message, 'name'):
            print(f"名称：{message.name}")

        print("\n\n执行流程：")
        print("""
            1. HumanMessage    → 用户问题
            2. AIMessage       → AI 决定调用工具（包含 tool_calls）
            3. ToolMessage     → 工具执行结果
            4. AIMessage       → AI 基于结果生成最终答案
            """)

        print("\n关键点：")
        print("  - Agent 自动完成这个循环")
        print("  - 所有步骤都记录在 messages 中")
        print("  - 最后一条消息是最终答案")



