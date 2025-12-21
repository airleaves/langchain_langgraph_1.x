"""
Langchain 1.0 -Simple Agent (使用 create_agent)
===============================================

本模块重点讲解:
1.使用create_agent 创建 Agent (Langchain 1.0 新API)
2.Agent 自动决定何时使用工具
3.Agent 执行循环的工作原理

    ⚠️ 重要更新：
- LangChain 1.0 中，Agent 创建使用 `create_agent`
- 它来自 `langchain.agents` 模块（LangChain 1.0 新增）
- 旧的 `create_react_agent`（langgraph.prebuilt）已弃用
"""
import os
import sys

# 添加父目录到路径以导入工具
parent_dir =os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(parent_dir,'04_custom_tools','tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent #Langchain 1.0 API
from langgraph.checkpoint.memory import MemorySaver #用于多轮对话

#导入自定义工具
from weather import get_weather
from calculator import calculator
from web_search import web_search

# 加载环境变量
load_dotenv()
KEY=os.getenv("DEEPSEEK_API_KEY")

# 加载模型
model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY,
)

# ============================================================================
#  示例1 :创建Agent
# ============================================================================
def example_1_create_agent():
    """
    示例1: 创建 Agent

    关键:
    1.使用create_agent 函数(langchain 1.0 API)
    2.传入model 和 tools
    3.Agent 会自动决定是否使用工具
    """
    print("\n" + "=" * 70)
    print("示例 1：创建第一个 Agent")
    print("=" * 70)

    #Langchain 1.0 新API: create_agent
    agent=create_agent(
        model=model,
        tools=[get_weather,calculator,web_search],
        system_prompt="""
        你是一个有帮助的助手，可以查询天气信息
        """
    )
    print("\nAgent 创建成功！")
    print("配置的工具：get_weather")
    print("使用 LangChain 1.0 API：create_agent")

    # 测试 需要工具的问题
    print("\n测试1：询问天气(需要工具)")
    response=agent.invoke({
        "messages":[
            {"role":"user","content":"北京今天天气怎么样？"}
        ]
    })
    print(f"\nAgent 回复：{response['messages'][-1].content}")
    print(response,"-=-=-=-=--=-=-========")
    # 测试：不需要工具的问题
    print("\n测试2：普通问题（不需要工具）")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "你好，介绍一下你自己"}]
    })

    print(f"\nAgent 回复：{response['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 自动判断是否需要使用工具")
    print("  - 需要工具时：调用工具 → 获取结果 → 生成回答")
    print("  - 不需要时：直接回答")

# =============================================================================
# 示例2 : 多工具 Agent
# =============================================================================
def example_2_multi_tools_agent():
    """
    示例2 : 配置多个工具 Agent

    Agent 会根据问题选择合适的工具
    """
    print("\n" + "=" * 70)
    print("示例 2：多工具 Agent")
    print("=" * 70)

    # 创建配置多个工具的 Agent
    agent=create_agent(
        model=model,
        tools=[get_weather,calculator,web_search],
        system_prompt="你是一个有帮助的助手。"
    )
    print("\n配置的工具：")
    print("  - get_weather（天气查询）")
    print("  - calculator（计算器）")
    print("  - web_search（网页搜索）")

    # 测试不同类型的问题
    tests = [
        "上海的天气怎么样？",  # 应该用 get_weather
        "15 乘以 23 等于多少？",  # 应该用 calculator
    ]
    for i,test in enumerate(tests):
        print(f"\n测试{i+1}：{test}")
        response = agent.invoke({
            "messages": [{"role": "user", "content": test}]
        })
        print(f"Agent 回复：{response['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 从多个工具中选择最合适的")
    print("  - 基于工具的 docstring 理解工具用途")

# =============================================================================
# 示例3 : 带系统的提示词
# =============================================================================
def example_3_agent_with_system_prompt():
    """
    示例3 : 自定义 Agent 的行为

    使用 prompt 参数(注意：不是system_prompt)
    """
    print("\n" + "=" * 70)
    print("示例 3：自定义 Agent 行为")
    print("=" * 70)

    # create_agent 使用 system_prompt 参数（字符串或 SystemMessage）
    system_messages ="""
    你是一个友好的助手。
    特点:
        - 回答简洁明了
        - 使用工具前先说明
        - 结果用表格或列表清晰展示
    """
    agent=create_agent(
        model=model,
        tools=[get_weather,calculator,web_search],
        system_prompt=system_messages,# 使用system_prompt参数
    )
    print("\n测试，自定义行为的Agent")
    response=agent.invoke({
        "messages":[
            {"role":"user","content":"上海的天气怎么样？"}
        ]
    })
    print(f"Agent 输出：{response['messages'][-1].content}")

    print("\n关键点：")
    print("  - system_prompt 参数定义 Agent 的系统提示")
    print("  - 可以指定输出格式、语气、工作流程等")
    print("  - 也可以传入 SystemMessage 对象")


# =============================================================================
# 示例4 : Agent执行过程详解
# =============================================================================
def exampple_4_agent_execution_process():
    """
    示例4 :  查看Agent 执行的完整流程

    理解 Agent 如何一步步工作
    """
    print("\n" + "=" * 70)
    print("示例 4：Agent 执行过程详解")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="你是一个专注于计算的助手。"
                      "工具里面调用的运算符是英文简写"
                        "列如：add、sub、mul、div"
    )
    print("\n问题：25 乘以 8 等于多少？")
    print("\nAgent 执行过程：")

    response = agent.invoke({
        "messages": [{"role": "user", "content": "25 乘以 8 等于多少？"}]
    })
    # 显示完整的消息历史
    print("\n完整消息历史：")
    for i, msg in enumerate(response['messages'], 1):
        print(f"\n--- 消息 {i} ({msg.__class__.__name__}) ---")
        if hasattr(msg, 'content'):
            print(f"内容：{msg.content}")
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"工具调用：{msg.tool_calls}")

# ==============================================================================
# 示例5 : 多轮对话 Agent(使用MemorySaver)
# =============================================================================
def example_5_multi_turn_agent():
    """
    示例5 : Agent 的多轮对话

    关键: 使用MemorySaver 保持对话历史
    """
    print("\n" + "=" * 70)
    print("示例 5：多轮对话 Agent")
    print("=" * 70)

    #创建内存检查点
    memory=MemorySaver()

    # 创建带记忆的Agent
    agent = create_agent(
            model=model,
            tools=[calculator,web_search,get_weather],
            system_prompt="你是一个有帮助的助手。",
            checkpointer=memory
        )

    #使用thread_id 来保持对话
    config={
        "configurable":{
            "thread_id":"conversation-1"
        }
    }

    # 第一轮
    response1=agent.invoke(
        {"messages":[{"role":"user","content":"10加5等于多少？"}]},
        config= config)
    print(f"Agent 回复：{response1['messages'][-1].content}")

    # 第二轮 继续上一轮的对话(记忆自动保持)
    reponse2=agent.invoke({
        "messages":[
            {"role":"user","content":"上海的天气怎么样？"}
        ]
    },config= config #使用相同的thread_id
    )
    print(f"Agent 输出：{reponse2['messages'][-1].content}")

    print("\n关键点：")
    print("  - 使用 MemorySaver 作为 checkpointer")
    print("  - 通过 thread_id 区分不同的对话")
    print("  - Agent 自动记住上下文")
    print("  - 不需要手动传递历史消息")

# =============================================================================
# 示例6 : Agent 最佳实践
# =============================================================================
def example_6_best_practices():
    """
    示例6：使用 Agent 的最佳实践
    """
    print("\n" + "="*70)
    print("示例 6：Agent 最佳实践")
    print("="*70)

    print("""
最佳实践：

1. 工具选择
   - 只给 Agent 需要的工具（工具太多会混淆）
   - 工具的 docstring 要清晰
   - 每个工具功能单一

2. System Prompt
   - 明确说明 Agent 的角色
   - 定义输出格式
   - 说明何时使用工具

3. 错误处理
   - 工具内部捕获异常
   - 返回友好的错误信息
   - Agent 可以处理工具失败

4. 性能优化
   - 减少不必要的工具调用
   - 缓存常用查询结果
   - 使用流式输出（后续学习）

5. 测试
   - 测试各种问题类型
   - 测试边界情况
   - 验证工具选择是否正确

6. API 注意事项（LangChain 1.0）
   - 使用 create_agent（LangChain 1.0 新 API）
   - LangChain 1.0 新 API
   - 使用 MemorySaver 实现多轮对话
    """)

    print("\n示例：良好配置的 Agent")

    agent = create_agent(
        model=model,
        tools=[get_weather, calculator],
        system_prompt="""你是一个专业的助手。
        工作流程：
        1. 仔细理解用户问题
        2. 如果需要工具，先说明将要做什么
        3. 调用工具获取准确信息
        4. 基于结果给出清晰答案
        
        输出要求：
        - 简洁明了
        - 数据准确
        - 格式清晰"""
            )

    print("\n测试：")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气如何？"}]
    })
    print(f"Agent 回复：{response['messages'][-1].content}")


# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - Simple Agent (create_agent)")
    print("="*70)

    try:
        # example_1_create_agent()
        # input("\n按 Enter 继续...")
        #
        # example_2_multi_tools_agent()
        # input("\n按 Enter 继续...")

        # example_3_agent_with_system_prompt()
        # input("\n按 Enter 继续...")

        # exampple_4_agent_execution_process()
        # input("\n按 Enter 继续...")

        example_5_multi_turn_agent()
        input("\n按 Enter 继续...")

        example_6_best_practices()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  ✅ create_agent 创建 Agent（LangChain 1.0）")
        print("  ✅ Agent 自动判断何时使用工具")
        print("  ✅ 执行循环：问题 → 工具调用 → 结果 → 回答")
        print("  ✅ 使用 MemorySaver 实现多轮对话")
        print("  ✅ prompt 参数定义 Agent 行为")
        print("\n下一步：")
        print("  06_agent_loop - 深入理解 Agent 执行循环")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()






#
# {'messages': [HumanMessage(content='北京今天天气怎么样？', additional_kwargs={}, response_metadata={}, id='00f69b1c-e6f4-4a10-96e2-5ceed5d6729b'),
#               AIMessage(content='我来帮您查询北京的天气信息。', additional_kwargs={'refusal': None},
#                         response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 548, 'total_tokens': 599, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 512}, 'prompt_cache_hit_tokens': 512, 'prompt_cache_miss_tokens': 36}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': 'd1b3d441-aeb4-4ab5-bbc2-e1dc756ae7c8', 'finish_reason': 'tool_calls', 'logprobs': None},
#                         id='lc_run--019b403e-94f4-72c1-b147-e7a1795f2cab-0', tool_calls=[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_00_2mPEkOnJg0LXBJzqXlHwqO22', 'type': 'tool_call'}],
#                         usage_metadata={'input_tokens': 548, 'output_tokens': 51, 'total_tokens': 599, 'input_token_details': {'cache_read': 512}, 'output_token_details': {}}),
#               ToolMessage(content='晴天，温度25度', name='get_weather', id='977a6ced-244a-41a8-9e2d-5c5bbe8f6d6d', tool_call_id='call_00_2mPEkOnJg0LXBJzqXlHwqO22'),
#               AIMessage(content='根据查询结果，北京今天的天气是晴天，温度25度。天气不错，适合外出活动！', additional_kwargs={'refusal': None},
#                         response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 621, 'total_tokens': 642, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 576}, 'prompt_cache_hit_tokens': 576, 'prompt_cache_miss_tokens': 45}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': 'c9ecfd00-139c-4861-9fc4-533609bd4eff', 'finish_reason': 'stop', 'logprobs': None},
#                         id='lc_run--019b403e-a009-7b72-bc63-08a727b760ab-0',
#                         usage_metadata={'input_tokens': 621, 'output_tokens': 21, 'total_tokens': 642, 'input_token_details': {'cache_read': 576}, 'output_token_details': {}})
#               ]}


















