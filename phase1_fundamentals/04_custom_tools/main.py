"""
Langchain 1.0 -自定义工具(@tool)
===============================

本模块重点讲解:
1.使用@tool 装饰器创建工具(Langchain 1.0 推荐方式)
2. 工具的参数和文档字符串的重要性
3.测试工具
"""
import os
import sys



# 将tools目录添加到Python路径中，以便能够导入自定义工具模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))


from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool



# 导入自定义工具
from tools.weather import get_weather
from tools.web_search import web_search
from tools.calculator import calculator

load_dotenv()


model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ============================================================================
# 示例1 : 使用@tool 装饰器创建工具
# ============================================================================
def example_1_simple_tool():
    """
    示例1: 使用@tool 装饰器创建工具

    关键：
    1. 使用 @tool 装饰器
    2. 必须有 docstring（文档字符串）
    3. 参数要有类型注解
    """
    @tool
    def get_current_time()->str:
        """
        获取当前时间

        返回:
            当前时间字符串
        """
        from  datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n工具名称:", get_current_time.name)
    print("工具描述:", get_current_time.description)
    print("工具参数:", get_current_time.args)

    # 调用工具
    result=get_current_time.invoke({})
    print("工具结果:", result)


# ============================================================================
# 示例2 : 带参数的工具
# ============================================================================
def example_2_tool_with_params():
    """
    示例2: 带参数的工具

    重点：参数的文档说明
    """
    print("\n" + "=" * 70)
    print("示例 2：带参数的工具")
    print("=" * 70)

    # 测试工具
    print("\n测试工具：")
    result1 = get_weather.invoke({"city": "北京"})
    print(f"北京天气: {result1}")

    result2 = get_weather.invoke({"city": "上海"})
    print(f"上海天气: {result2}")

    print("\n💡 docstring 格式：")
    print('''
        @tool
        def my_tool(param1: str) -> str:
            """
            工具的简短描述

            参数:
                param1: 参数说明

            返回:
                返回值说明
            """
        ''')

# ============================================================================
# 示例3 : 多参数工具
# =================================================================
def example_3_multiple_params():
    """
    示例3: 多参数工具
    """
    print("\n" + "="*70)
    print("示例 3：多参数工具 - 计算器")
    print("="*70)

    print("\n计算器工具信息：")
    print(f"名称: {calculator.name}")
    print(f"描述: {calculator.description}")

    # 测试
    print("\n测试工具：")
    test1=[
        {"operation": "add", "a": 1, "b": 2},
        {"operation": "sub", "a": 5, "b": 3},
        {"operation": "mul", "a": 4, "b": 6},
        {"operation": "div", "a": 10, "b": 2},
    ]

    for test in test1:
        result = calculator.invoke(test)
        print(f"{test['operation']} {test['a']} and {test['b']} = {result}")


# ============================================================================
# 示例4 : 可选参数
# ============================================================================
def example_4_optional_params():
    """
    示例4: 可选参数

    使用Optional[类型] 和默认值
    """
    print("\n" + "=" * 70)
    print("示例 4：可选参数 - 搜索工具")
    print("=" * 70)

    #使用默认参数
    print("\n使用默认参数：")
    result1 = web_search.invoke({"query": "Python"})
    print(f"搜索结果：{result1}")

    #使用可选参数
    print("\n使用可选参数：")
    result2 = web_search.invoke({"query": "LangChain", "num_results": 2})
    print(f"搜索结果：{result2}")

#=============================================================================
# 示例5 : 工具绑定到模型
#=============================================================================
def example_5_bind_tools():
    """
    示例5： 将工具绑定到模型

    这是让AI 使用工具的第一步
    """
    print("\n" + "=" * 70)
    print("示例 5：将工具绑定到模型")
    print("=" * 70)

    #绑定工具到模型
    model_with_tools=model.bind_tools([get_weather,calculator])

    print("模型已经绑定工具")
    print("- get_weather")
    print("- calculator")

    # 调用模型(模型可以选择使用工具)
    print("\n测试：AI 是否会调用天气工具？")
    response=model_with_tools.invoke("北京今天的天气怎么样？")
    print(f"模型输出：{response}")

    #检查模型是否要求调用工具
    if response.tool_calls:
        print(f"\n✅ AI 决定使用工具！")
        print(f"工具调用: {response.tool_calls}")
    else:
        print(f"\nℹ️ AI 直接回答（未使用工具）")
        print(f"回复: {response.content}")

    print("\n💡 下一步：")
    print("  在 05_simple_agent 中，我们将学习如何让 AI 自动执行工具")


#=============================================================================
# 示例6 : 工具的最佳实践
#=============================================================================
def example_6_best_practices():
    """
    示例6： 工具的最佳实践
    """
    print("\n" + "=" * 70)
    print("示例 6：工具开发最佳实践")
    print("=" * 70)
    print("\n✅ 好的工具设计：")
    print("""
    1. 清晰的 docstring
       @tool
       def search_products(query: str, max_results: int = 10) -> str:
           '''
           在产品数据库中搜索产品

           参数:
               query: 搜索关键词
               max_results: 最大返回数量，默认10

           返回:
               产品列表的JSON字符串
           '''

    2. 明确的参数类型
       - 使用类型注解：str, int, float, bool
       - 可选参数用 Optional[类型]

    3. 返回字符串
       - 工具应该返回 str（AI 最容易理解）
       - 复杂数据可以返回 JSON 字符串

    4. 错误处理
       - 在工具内部捕获异常
       - 返回友好的错误消息

    5. 功能单一
       - 一个工具做一件事
       - 不要把多个功能塞进一个工具
        """)

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - 自定义工具")
    print("=" * 70)

    try:
        # example_1_simple_tool()
        # input("\n按 Enter 继续...")
        #
        # example_2_tool_with_params()
        # input("\n按 Enter 继续...")
        #
        # example_3_multiple_params()
        # input("\n按 Enter 继续...")
        #
        # example_4_optional_params()
        # input("\n按 Enter 继续...")

        example_5_bind_tools()
        input("\n按 Enter 继续...")

        example_6_best_practices()
        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  ✅ 使用 @tool 装饰器创建工具")
        print("  ✅ 必须有清晰的 docstring")
        print("  ✅ 参数要有类型注解")
        print("  ✅ 工具返回字符串")
        print("\n下一步：")
        print("  05_simple_agent - 学习如何让 AI 自动使用工具")

    except Exception as e:
        print(f"\n❌ 错误：{e}")


if __name__ == "__main__":
    main()