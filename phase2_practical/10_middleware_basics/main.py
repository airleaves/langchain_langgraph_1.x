"""
Langchain 1.0 -Middleware Basics (中间件基础)
===========================================

本模块重点讲解:
1.什么是中间件(Middleware)
2.before_model 和 after_model 钩子
3.自定义中间件的创建
4.多个中间件的组合
5.内置中间件的使用
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware.types import StateT
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


# 加载环境变量
load_dotenv()

KEY=os.getenv("DEEPSEEK_API_KEY")

# 初始化模型
model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY,
)

# 创建工具
@tool
def get_weather(city: str)->str:
    """查询城市天气"""
    weather_data={"北京": "晴天", "上海": "阴天"}
    return weather_data.get(city, "未知城市")

# ==============================================================================
# 示例1:最简单的中间件
"""
关键规则:
1.必须继承AgentMiddleware <- 这个固定
2.方法名固定(before_model,after_model) <-这个固定
3.类名随意 <- 这个不固定

LangGraph 只看:
- 是否继承AgentMiddleware?
- 是否有before_model / after_model方法？
"""
# ===============================================================================
class LoggingMiddleware(AgentMiddleware): # 类名随意
    """
    日志中间件 - 记录每次模型调用

    before_model: 模型调用前执行
    after_model: 模型响应后执行
    """
    def before_model(self, state: StateT, runtime):
        """模型调用前"""
        print("\n[中间件] before_model: 准备调用模型")
        print(f"[中间件] 当前消息数: {len(state.get('messages', []))}")
        return None # 返回None 表示继续执行正常流程

    def after_model(self, state: StateT, runtime):
        """模型调用后"""
        print("[中间件] after_model: 模型已响应")
        last_message = state.get('messages', [])[-1]
        print(f"[中间件] 响应类型: {last_message.__class__.__name__}")
        return None  # 返回 None 表示不修改状态


def example_1_basic_middleware():
    """
    示例1 : 基础中间件 - 日志记录

    展示 before_model 和 after_model 的基本用法
    """
    print("\n" + "=" * 70)
    print("示例 1：基础中间件 - 日志记录")
    print("=" * 70)

    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[LoggingMiddleware()] # 添加中间件
    )
    print("\n 用户:你好")
    response=agent.invoke({
        "messages":[
            {"role": "user", "content": "你好"}
        ]
    })
    print(f"Agent 回复: {response['messages'][-1].content}")
    print("\n关键点：")
    print("  - before_model 在模型调用前执行")
    print("  - after_model 在模型响应后执行")
    print("  - 返回 None 表示不修改状态")

# ==============================================================================
# 示例2: 修改状态的中间件
# ================================================================================
class CallCountMiddleware(AgentMiddleware):
    """
    调用计数中间件 - 记录模型调用次数

    before_model: 模型调用前执行
    after_model: 模型响应后执行
    """

    def __init__(self):
        super().__init__()
        self.count=0 #简单计数器

    def after_model(self, state: StateT, runtime):
        """模型调用后"""
        self.count+=1
        print(f"[中间件] 模型调用次数: {self.count}")
        return None # 不修改 state

def example_2_state_modification():
    """
    示例2:中间件内部状态 -计数器

    展示如何在中间件内部维护状态(不依赖Agent state)
    """
    print("\n" + "=" * 70)
    print("示例 2：中间件内部状态 - 模型调用计数")
    print("=" * 70)

    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[CallCountMiddleware()],
        checkpointer=InMemorySaver()
    )

    config={"configurable": {"thread_id": "user_123"}}

    print("\n 第一次调用")
    agent.invoke({
        "messages":[
            {"role": "user", "content": "你好"}
        ]
    },config=config)
    print("\n 第二次调用")
    agent.invoke({
        "messages":[
            {"role": "user", "content": "你是谁"}
        ]
    },config=config)
    print("\n 第三次调用")
    agent.invoke({
        "messages":[
            {"role": "user", "content": "谢谢你 宝贝！"}
        ]
    },config=config)

    print("\n关键点：")
    print("  - 中间件内部维护计数器（self.count）")
    print("  - 不依赖 Agent state（更可靠）")
    print("  - 返回 None 表示不修改 Agent 状态")

# ===============================================================================
# 示例3: 消息修剪中间件
# ================================================================================
class MessagesTrimMiddleware(AgentMiddleware):
    """
    消息修剪中间件 - 限制消息数量

    before_model 修改消息列表
    注意 : 需要配合无checkpoint使用，否则历史会被恢复
    """
    def __init__(self, max_messages=5):
        super().__init__()
        self.max_messages=max_messages
        self.trimmed_count=0 # 统计修剪次数

    def before_model(self, state, runtime):
        """模型调用前，修剪消息"""
        messages=state.get("messages", [])

        if len(messages)>self.max_messages:
            # 保留最近的N条消息
            trimmed_messages=messages[-self.max_messages:]
            self.trimmed_count+=1
            print(f"\n [修剪] 消息从{len(messages)}减少到{len(trimmed_messages)}条")
            return {"messages": trimmed_messages}
        return None

def example_3_message_trimming():
    """
    示例3: 消息修剪中间件

    展示如何使用中间件来限制消息数量
    重点: 手动累计消息，观察修剪效果
    """
    print("\n" + "=" * 70)
    print("示例 3：消息修剪中间件 -限制消息数量")
    print("=" * 70)

    print("\n [说明] 不使用 checkpointer,手动管理消息历史\n")

    middleware=MessagesTrimMiddleware(max_messages=4) #最多保留4条
    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[middleware],
        # checkpointer=InMemorySaver() 不使用checkpointer
    )
    # 手动管理消息历史
    messages=[]
    for i in range(6):
        print(f"\n ----第{i+1}次对话----")

        # 新增用户消息
        new_msg={"role": "user", "content": f"这是第{i+1}条消息:简要回复"}
        messages.append(new_msg)

        print(f"调用前消息数:{len(messages)}")

        # 调用agent(middleware会修剪)
        response=agent.invoke({
            "messages":messages
        })
        messages=response['messages']

        print(f"调用后消息数:{len(messages)}")
        if len(messages)<=4:
            print(f"消息列表{[m.content[:15] for m in messages]}")

        print(f"\n修剪统计: 共修剪了 {middleware.trimmed_count} 次")

    print("\n关键点：")
    print("  - before_model 在传给模型前修剪消息")
    print("  - max_messages=4 限制发送给模型的消息数")
    print("  - 但返回的 response 会包含新生成的消息")
    print("  - 不使用 checkpointer 避免历史恢复")
    print("\n生产建议：")
    print("  - 简单修剪用这种方式")
    print("  - 复杂场景用 SummarizationMiddleware（第8章）")

# ===============================================================================
# 示例4: 输出验证中间件
# ================================================================================
class OutputVerificationMiddleware(AgentMiddleware):
    """
    输出验证中间件 - 检查响应长度

    after_model 修改模型输出
    """
    def __init__(self, max_length=100):
        super().__init__()
        self.max_length=max_length

    def after_model(self, state, runtime):
        """模型调用后，验证输出长度"""
        response=state.get('messages',[])
        if not response:
            return None

        last_message=response[-1]
        content=getattr(last_message,'content', '')
        if len(content)>self.max_length:
            print(f"\n[警告] 响应时间过长({len(content)}字符)，已截断到{self.max_length}")
            # 这里可以实现截断或重试逻辑

        return None
def example_4_output_validation():
    """
    示例4 : 输出验证 - 检查响应质量
    展示如何验证模型输出
    """
    print("\n" + "=" * 70)
    print("示例 4：输出验证 - 检查响应质量")
    print("=" * 70)

    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[OutputVerificationMiddleware(max_length=100)]
    )
    print("\n用户: 请详细介绍Python编程语言的历史、特点和应用")
    response=agent.invoke({
        "messages":[
            {"role": "user", "content": "请详细介绍Python编程语言的历史、特点和应用"}
        ]
    })
    print(f"\n助手: {response['messages'][-1].content[:100]}...")
    print("\n关键点：")
    print("  - after_model 可以验证输出")
    print("  - 可以实现重试、截断等逻辑")
    print("  - 保证输出质量")

# ================================================================================
# 示例5: 多个中间件组合
# ================================================================================
class TimingMiddleware(AgentMiddleware):
    """
    计时中间件 - 测量模型调用时间

    after_model 计时
    """
    def before_model(self, state, runtime):
        """模型调用前，记录开始时间"""
        print("\n[计时] 开始调用模型...")
        return None

    def after_model(self, state, runtime):
        """模型调用后，记录结束时间"""
        print("[计时] 模型调用完成")
        return None

def example_5_multiple_middleware():
    """
    示例5: 多个中间件组合
    展示中间件的执行顺序：
    - before_model: 正序（1→2→3）
    - after_model: 逆序（3→2→1）
    """
    print("\n" + "=" * 70)
    print("示例 5：多个中间件组合 -执行顺序")
    print("=" * 70)

    class Middleware1(AgentMiddleware):
        """中间件1"""
        def before_model(self, state, runtime):
            print("[中间件1] before_model")
            return None

        def after_model(self, state, runtime):
            print("[中间件1] after_model")
            return None

    class Middleware2(AgentMiddleware):
        """中间件2"""
        def before_model(self, state, runtime):
            print("[中间件2] before_model")
            return None

        def after_model(self, state, runtime):
            print("[中间件2] after_model")
            return None

    class Middleware3(AgentMiddleware):
        """中间件3"""
        def before_model(self, state, runtime):
            print("[中间件3] before_model")
            return None

        def after_model(self, state, runtime):
            print("[中间件3] after_model")
            return None

    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[
            Middleware1(),
            Middleware2(),
            Middleware3(),
            TimingMiddleware()
        ]
    )
    print("\n 执行一次调用，观察顺序:")
    agent.invoke({
        "messages":[
            {"role": "user", "content": "测试"}
        ]
    })
    print("\n关键点：")
    print("  - before_model: 正序执行（1→2→3）")
    print("  - after_model: 逆序执行（3→2→1）")
    print("  - 类似洋葱模型：1→2→3→模型→3→2→1")
# ================================================================================
# 示例6: 条件跳转
# ================================================================================
class MaxCallsMiddleware(AgentMiddleware):
    """
    最大调用限制中间件

    通过抛出异常来阻止模型调用(更可靠的方式)
    """
    def __init__(self,max_calls=5):
        super().__init__()
        self.max_calls=max_calls
        self.count=0 #简单计数器

    def before_model(self, state, runtime):
        """检查调用次数，超过限制则抛出异常"""
        if self.count>=self.max_calls:
            print(f"\n[限制] 已达到最大调用次数({self.max_calls})，停止调用")
            # 抛出自定义异常来阻止继续执行
            raise ValueError(f"已经达到最大调用次数限制:{self.max_calls}")
        print(f"[限制] 当前调用次数:{self.count}/{self.max_calls}")
        return None

    def after_model(self, state, runtime):
        """增加计数"""
        self.count += 1
        print(f"次数+1,{self.count}===")
        return None
def example_6_conditional_jump():
    """
    示例6 : 调用限制 - 通过异常阻止调用

    展示如何使用异常来阻止模型调用(比jump_to 更可靠)
    """

    print("\n"+ "="*70)
    print("示例 6：调用限制 - 最大调用次数")
    print("="*70)

    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[MaxCallsMiddleware(max_calls=3)],
        checkpointer=InMemorySaver()
    )
    config={"configurable":{"thread_id":"limit_test"}}

    for i in range(4):
        print(f"\n第{i+1}次调用:")
        try:
            response=agent.invoke({
                "messages":[
                    {"role": "user", "content": f"消息{i+1}"}
                ]
            },config)
            if response['messages']:
                print(f"  内容: {response['messages'][-1].content[:50]}...")
        except ValueError as e:
            print(f"  已阻止: {e}")
        except Exception as e:
            print(f"  调用失败: {e}")

    print("\n关键点：")
    print("  - before_model 中抛出异常可以阻止模型调用")
    print("  - 比 jump_to 更可靠（在 LangChain 1.0 中）")
    print("  - 中间件内部维护计数（self.count）")
    print("  - 用于实现熔断、限流等逻辑")
    print("\n注意：")
    print("  - jump_to 在 middleware 中可能不按预期工作")
    print("  - 推荐用异常来实现流程控制")

# ================================================================================
# 示例7: 内置中间件使用
# ================================================================================
def example_7_builtin_middleware():
    """
    示例7: 内置中间件 - SummarizationMiddleware

    使用 Langchain 提供的内置中间件
    """
    print("\n"+ "="*70)
    print("示例 7：内置中间件 - 自动摘要")
    print("="*70)

    from langchain.agents.middleware import SummarizationMiddleware

    agent=create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个 helpful 的助手。",
        middleware=[
            SummarizationMiddleware(
                model=model,
                # max_tokens_before_summary=800,
                trigger=('tokens', 200)
            )
        ],
        checkpointer=InMemorySaver()
    )

    config={"configurable":{"thread_id":"summary_test"}}

    # 连续多轮对话
    conversations = [
        "介绍一下 Python",
        "它有哪些特点？",
        "主要应用在哪些领域？",
        "和 Java 的区别是什么？"
    ]
    for i, conversation in enumerate(conversations,1):
        print(f"\n第{i}轮对话:")
        print(f"  问题: {conversation}")
        response=agent.invoke({
            "messages":[
                {"role": "user", "content": conversation}
            ]
        },config=config)
        print(f"  内容: {response['messages'][-1].content[:100]}...")
        print(f"总消息数:{len(response['messages'])}")

    print("\n关键点：")
    print("  - SummarizationMiddleware 自动摘要旧消息")
    print("  - 防止消息历史无限增长")
    print("  - 第 08 章详细学习过")

def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - Middleware Basics (中间件)")
    print("=" * 70)

    try:
        example_1_basic_middleware()
        input("\n按 Enter 继续...")
        example_2_state_modification()
        input("\n按 Enter 继续...")
        example_3_message_trimming()
        input("\n按 Enter 继续...")
        example_4_output_validation()
        input("\n按 Enter 继续...")
        example_5_multiple_middleware()
        input("\n按 Enter 继续...")
        example_6_conditional_jump()
        input("\n按 Enter 继续...")
        example_7_builtin_middleware()

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()