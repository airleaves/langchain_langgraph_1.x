"""
Langchain 1.0 - Structured Output (结构化输出)
=============================================

本模块重点讲解:
1.使用Pydantic 定义输出模式
2.with_structured_output() 方法
3. 嵌套模型和复杂结构
4.枚举类型和验证
5.实际应用场景

注意 :
- with_structured_output() 可能在某些模型上不完全支持
- 如遇到错误，本模块会自动使用JSON 解析作为fallback
"""
import os
import json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel,Field
from typing import Optional,List,TypeVar,Type
from enum import Enum

# 加载环境变量
load_dotenv()

KEY=os.getenv("DEEPSEEK_API_KEY")

# 初始化模型
model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY
)

# ================辅助函数====================

T=TypeVar("T", bound=BaseModel)

def safe_structured_ouput(prompt:str,output_class:Type[T])->T:
    """
    安全的结构化输出函数

    先尝试 with_structured_output,失败则使用 JSON 解析 fallback
    """

    # 尝试使用 with_structured_output
    try:
        structured_llm=create_safe_structured_llm(output_class)
        result= structured_llm.inovke(prompt)
        return result
    except Exception as e:
        print(f"with_structured_output() 错误: {e}")

    #Fallback : 手动 JSON 解析
    json_prompt=f"""
        {prompt}
        请严格按照以下JSON 格式返回(不要添加任何其他内容):
        {json.dumps(output_class.model_json_schema().get('properties',{}),indent=2,ensure_ascii=False)}
        只返回JSON，不要其他文字。
        """
    response=model.invoke([HumanMessage(content=json_prompt)])
    content=response.content.strip()

    # 清理Markdown 格式
    if "```json" in content:
        content=content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content=content.split("```")[1].split("```")[0]

    try:
        data=json.loads(content.strip())
        return output_class.model_validate(data)
    except Exception as e:
        print(f"JSON 解析错误: {e}")
        raise ValueError("无法解析结构化输出")



def create_safe_structured_llm(output_class):
    """创建带fallback 的结构化输出LLM"""
    base_llm=model.with_structured_output(output_class)

    class SafeStructuredLLM(BaseModel):
        def inovke(self,prompt):
            try:
                return base_llm.invoke(prompt)
            except Exception as e:
                print(f"with_structured_output() 失败，使用 JSON 解析 fallback{e}")
                return safe_structured_ouput(prompt,output_class)

    return SafeStructuredLLM()

# ===============================================
# 示例1 : 基础结构化输出
# ===============================================
class Person(BaseModel):
    """人物信息"""
    name:str =Field(description="姓名")
    age:int =Field(description="年龄")
    occupation:str =Field(description="职业")

def example_1_basic_structured_output():
    """
    示例1 : 基础结构化输出

    使用 with_structured_output() 将LLM 输出转为 Pydantic 对象
    """
    print("\n" + "=" * 70)
    print("示例 1：基础结构化输出 - Pydantic 模型")
    print("=" * 70)

    print("\n 提示:张三是一名 30岁的软件工程师")

    # 使用安全的结构化输出函数
    result=safe_structured_ouput("张三是一名 30岁的软件工程师",Person)

    print(f"\n 返回类型:{type(result)}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"职业: {result.occupation}")

    print("\n关键点：")
    print("  - with_structured_output(Person) 返回 Person 对象")
    print("  - 不需要手动解析 JSON")
    print("  - 自动类型验证（age 必须是 int）")

# ===============================================
# 示例2 : 提取多个对象(列表)
# ===============================================
class Book(BaseModel):
    """书籍信息"""
    title:str=Field(description="书名")
    author:str=Field(description="作者")
    year:int=Field(description="出版年份")

class BookList(BaseModel):
    """书籍列表"""
    books:List[Book]=Field(description="书籍列表")

def example_2_list_extraction():
    """
    示例2: 提取多个对象

    从文本中提取多个结构化对象
    """
    print("\n" + "=" * 70)
    print("示例 2：提取多个对象 - 列表")
    print("=" * 70)

    structured_llm=create_safe_structured_llm(BookList)

    text="""
     《三体》是刘慈欣 2008 年的科幻小说。
    《流浪地球》也是刘慈欣的作品，2000 年出版。
    《北京折叠》是郝景芳 2012 年的小说。
    """

    print(f"\n文本:{text.strip()}")
    result=structured_llm.inovke(f"从以下文本提取所有书籍信息:\n{text}")
    print(f"\n返回类型:{type(result)}")

    print(f"\n提取到{len(result.books)}本书：")
    for i,book in enumerate(result.books,1):
        print(f"{i}. {book.title} by {book.author} ({book.year})")

    print("\n关键点：")
    print("  - books: List[Book] 定义列表类型")
    print("  - LLM 自动识别并提取多个对象")
    print("  - 返回的是 Python 列表，可直接遍历")

# ===============================================
# 示例3 : 嵌套模型
# ===============================================
class Address(BaseModel):
    """地址"""
    city:str=Field(description="城市")
    district:str=Field(description="区县")
class Company(BaseModel):
    """公司信息"""
    name:str=Field(description="公司名称")
    employee_count:int=Field(description="员工数量")
    address:Address=Field(description="公司地址")

def example_3_nested_model():
    """
    示例3: 嵌套模型

    使用嵌套模型处理结构化数据
    """
    print("\n" + "=" * 70)
    print("示例 3：嵌套模型 - 复杂结构")
    print("=" * 70)

    structured_llm=create_safe_structured_llm(Company)

    print("\n提示: 阿里巴巴公司在杭州滨江区，有约 10 万名员工")
    result = structured_llm.inovke("阿里巴巴公司在杭州滨江区，有约 10 万名员工")

    print(f"\n 公司名称: {result.name}")
    print(f" 员工数量: {result.employee_count}")
    print(f" 公司地址: {result.address.city} {result.address.district}")

    print("\n关键点：")
    print("  - Address 嵌套在 Company 中")
    print("  - LLM 自动识别层级关系")
    print("  - 通过 result.address.city 访问嵌套字段")

# ===============================================
# 示例4 : 可选字段和默认值
# ===============================================
class Product(BaseModel):
    """产品信息"""
    name: str =Field(description="产品名称")
    price: float =Field(description="价格")
    description: Optional[str] =Field(None,description="描述(可选)")
    stock:int =Field(100,description="库存数量(默认 100)")

def example_4_optional_and_defaults():
    """
    示例4:可选字段和默认值

    处理不完整的信息
    """
    print("\n" + "=" * 70)
    print("示例 4：可选字段和默认值")
    print("=" * 70)

    structured_llm=create_safe_structured_llm(Product)
    print("\n场景1: 完整信息")
    result1=structured_llm.inovke("产品名称: 苹果手机\n价格: 6999.99\n 描述:最新款智能手机\n库存:109台")
    print(f"\n产品名称: {result1.name}")
    print(f"价格: {result1.price}")
    print(f"描述: {result1.description}")
    print(f"库存: {result1.stock}")

    print("\n场景2: 缺少描述字段")
    result2=structured_llm.inovke("MackBook Pro 售价12999元")
    print(f"\n产品名称: {result2.name}")
    print(f"价格: {result2.price}")
    print(f"描述: {result2.description}")
    print(f"库存: {result2.stock}")

    print("\n关键点：")
    print("  - Optional[str] 表示字段可以为 None")
    print("  - Field(100, ...) 设置默认值")
    print("  - LLM 未提供的信息会使用默认值")

# ===============================================
# 示例5 :枚举类型
# ===============================================
class Priority(str,Enum):
    """优先级"""
    HIGH="HIGH"
    MEDIUM="MEDIUM"
    LOW="LOW"

class Task(BaseModel):
    """任务信息"""
    title:str=Field(description="任务标题")
    description:str=Field(description="任务描述")
    priority:Priority=Field(description="优先级")

def example_5_enum_types():
    """
    示例5: 枚举类型
    限制字段的可选值
    """
    print("\n" + "=" * 70)
    print("示例 5：枚举类型 -限制可选值")
    print("=" * 70)
    structured_llm=create_safe_structured_llm(Task)
    print("\n提示: 完成季度报告，这是紧急任务")
    result = structured_llm.inovke("完成季度报告，这是紧急任务")

    print(f"\n任务标题: {result.title}")
    print(f"任务描述: {result.description}")
    print(f"优先级: {result.priority.value}")

    print("\n关键点：")
    print("  - Priority(str, Enum) 定义枚举")
    print("  - LLM 只能选择 LOW/MEDIUM/HIGH")
    print("  - 自动验证，无效值会报错")

# ===============================================
# 示例6 : 实际应用 - 客户端信息提醒
# ===============================================
class CustomerInfo(BaseModel):
    """客户信息"""
    name:str=Field(description="客户姓名")
    phone:str=Field(description="手机号码")
    email:Optional[ str]=Field(None,description="邮箱地址")
    issue:str =Field(description="问题描述")
    urgency:Priority=Field(description="紧急程度")

def exampele_6_customer_info_extraction():
    """
    示例6: 实际应用 - 客户信息提醒

    从客服对话中提取结构化信息
    """
    print("\n" + "=" * 70)
    print("示例 6：实际应用 - 客户信息提取")
    print("=" * 70)
    structured_llm=create_safe_structured_llm(CustomerInfo)

    conversation = """
        客服: 您好，请问有什么可以帮助您？
        客户: 我是李明，电话 138-1234-5678，我的订单一直没发货，很着急！
        客服: 好的，我帮您查一下
        """
    print(f"\n 对话记录\n{conversation} ")

    result=structured_llm.inovke(f"从以下客服对话中提取客户信息：\n{conversation}")
    print(f"\n客户姓名: {result.name}")
    print(f"手机号码: {result.phone}")
    print(f"邮箱地址: {result.email}")
    print(f"问题描述: {result.issue}")
    print(f"紧急程度: {result.urgency.value}")

    print("\n应用场景：")
    print("  - 自动填充 CRM 系统")
    print("  - 工单自动分类")
    print("  - 紧急问题优先处理")

# ===============================================
# 示例7: 实际应用 - 产品评论分析
# ===============================================
class Sentiment(str, Enum):
    """情感"""
    POSITIVE = "正面",
    NEGATIVE = "负面"
    NEUTRAL = "中立"

class Review(BaseModel):
    """评论"""
    product: str = Field(description="产品名称")
    rating: int = Field(description="评分1-5")
    sentiment: Sentiment = Field(description="情感倾向")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")

def exampele_7_product_review_analysis():
    """
    示例7: 实际应用 - 产品评论分析

    从产品评论中提取结构化信息
    """
    print("\n" + "=" * 70)
    print("示例 7：实际应用 - 产品评论分析")
    print("=" * 70)

    structured_llm = create_safe_structured_llm(Review)

    review_text = """
        这款 iPhone 15 Pro 真的很不错！摄像头非常强大，夜拍效果惊艳。
        钛金属边框手感也很好。但是价格有点贵，而且没有充电器。
        总体来说还是值得购买的，我给 4 分。
        """

    print(f"\n评论内容:\n{review_text}")
    result = structured_llm.inovke(f"分析以下产品评论：\n{review_text}")

    print("\n分析结果：")
    print(f"  产品: {result.product}")
    print(f"  评分: {result.rating} / 5")
    print(f"  情感: {result.sentiment.value}")
    print(f"  优点: {', '.join(result.pros)}")
    print(f"  缺点: {', '.join(result.cons)}")

    print("\n应用场景：")
    print("  - 批量处理用户评论")
    print("  - 自动生成分析报告")
    print("  - 发现产品改进点")

def main():
    print("\n" + "=" * 70)
    print(" LangChain 1.0 - Structured Output (结构化输出)")
    print("=" * 70)

    try:
        # example_1_basic_structured_output()
        # input("按 Enter 键继续...")
        # example_2_list_extraction()
        # input("按 Enter 键继续...")
        # example_3_nested_model()
        # input("按 Enter 键继续...")
        # example_4_optional_and_defaults()
        # input("按 Enter 键继续...")
        example_5_enum_types()
        input("按 Enter 键继续...")
        exampele_6_customer_info_extraction()
        # input("按 Enter 键继续...")
        # exampele_7_product_review_analysis()


    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()