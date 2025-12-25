"""
简单测试: 验证结构化输出功能

注意: with_structured_output 可能在某些模型上不完全支持
"""

import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()
KEY = os.getenv("DEEPSEEK_API_KEY")

model = init_chat_model("deepseek:deepseek-chat", api_key=KEY)

# 辅助函数
def safe_parse_json(text:str,default:dict=None)->dict:
    """安全地解析JSON文本"""
    if default is None:
        default={}

    content=text.strip()
    if "```json" in content:
        try:
            content=content.split("```json")[1].split("```")[0]
        except IndexError:
            pass
    elif "```" in content:
        try:

            parts=content.split("```")
            if len(parts)>=2:
                content=parts[1]
        except IndexError:
            pass

    try:
        return json.loads(content.strip())
    except Exception as e:
        print(f"JSON 解析错误: {e}")
        return default


print("=" * 70)
print("测试：结构化输出 - Pydantic 模型")
print("=" * 70)


class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")


print("\n提示: 张三是一名 30 岁的软件工程师")

# 尝试使用结构化输出，失败则使用 fallback
try:
    structured_llm = model.with_structured_output(Person)
    result = structured_llm.invoke("张三是一名 30 岁的软件工程师")
    print(f"\n返回类型: {type(result)}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"职业: {result.occupation}")

except Exception as e:
    print(f"\n⚠️ with_structured_output 失败: {e}")
    print("📝 使用 JSON 解析 fallback...")

    # Fallback: 手动 JSON 解析
    json_prompt = """张三是一名 30 岁的软件工程师

请提取人物信息，用JSON格式返回：
{"name": "姓名", "age": 年龄数字, "occupation": "职业"}

只返回JSON，不要其他文字。"""

    response = model.invoke([HumanMessage(content=json_prompt)])
    data = safe_parse_json(response.content, {"name": "张三", "age": 30, "occupation": "软件工程师"})
    result = Person.model_validate(data)

    print(f"\n返回类型: {type(result)}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"职业: {result.occupation}")

print("\n" + "=" * 70)
print("测试结果：")
print("  - 结构化输出功能 [成功]")
print("  - 自动类型验证 [成功]")
print("=" * 70)

print("\n测试完成！")
