from posthog import api_keyfrom pywin.framework.toolmenu import toolsfrom pywin.framework.toolmenu import toolsfrom pywin.framework.toolmenu import tools

# 04 - Custom Tools (自定义工具)

## 核心概念

**工具(Tool) = 给AI的函数**

使用`@tool`装饰器 ，让AI能调用你的Python函数

## @tool 基本用法

```python
from langchain_core.tools import tool

@tool
def get_weather(city:str)->str:
    """
    获取指定城市的天气信息
    
    参数:
    city: 城市名称
    返回:
    天气信息字符串
    """
    return "晴天，温度15℃"
```

### 关键要点
|必须项| 说明                |
|------|-------------------|
|`@tool`装饰器| 声明这是一个工具|
|**docstring**| AI读这个来理解工具用途，非常重要 |
|类型注解|参数和返回值的类型|
|返回`str`|工具应该返回字符串(AI最容易理解)|

## 工具的 docstring

**AI 依赖 docstring 来理解工具用途**

```python
@tool
def my_tool(param:str)->str:
    """
    工具的简短描述(AI 读这个！)
    
    参数:
    param: 参数描述
    返回:
    返回值描述
    """
```

## 参数类型

### 1.单参数
```python
@tool
def get_weather(city:str)->str:
    """获取指定城市的天气信息"""

```
### 2.多个参数
```python
@tool
def calculator(operator:str, a:int, b:int)->str:
    """
     执行数学计算
     参数:
     operator: 操作符 add / subtract / multiply / divide
     a: 数字
     b: 数字
     """
    ...
```

### 3.可选参数
```python
from typing import Optional
@tool
def web_search(query:str, num_results:Optional[int]=5)->str:
    """
    搜索网页
    参数:
    query: 搜索内容
    num_results: 结果数量
    """
    ...
```

## 调用工具

工具有两种调用方式：

### 1.直接调用(测试用)
```python
# 使用invoke()方法
result =get_weather.invoke("北京")
print(result) # 晴天，温度15℃
```
### 2.绑定到模型(让AI调用)
```python
from langchain.chat_models import init_chat_model
model = init_chat_model(model="deepseek:deepseek-chat",api_key="")

#绑定工具
model_with_tools =model.bind_tools([get_weather])

#AI可以决定是否调用工具
response=model_with_tools.invoke("北京的天气如何？")

#检查AI是否要调式工具

if response.tool_calls:
    print("AI 想调用工具",response.tool_calls)
else:
    print("AI 直接回答",response.content)
```
## 工具属性

创建工具后,可以查看其属性：
```python
@tool
def my_tool(param:str)->str:
    """工具描述"""
print(my_tool.name)
print(my_tool.description)
print(my_tool.args)
```

## 最佳实战

### 1.清晰的描述
```python
# 好
@tool
def search_flights(origin:str, destination:str, date:str)->str:
    """
    搜索航班信息
    参数:
    origin: 出发城市
    destination: 目标城市
    date: 航班日期
    返回:
    可用航班的JSON列表
    """
```
### 2. 功能单一
```python
# ❌ 不好：一个工具做太多事
@tool
def do_everything(action: str, data: str) -> str:
    """做各种事情"""
    if action == "weather": ...
    elif action == "calculate": ...
    elif action == "search": ...

# ✅ 好：每个工具做一件事
@tool
def get_weather(city: str) -> str:
    """获取天气"""
    ...

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """计算"""
    ...
```

### 3. 错误处理
```python
@tool
def divide(a: float, b: float) -> str:
    """
    除法计算

    参数:
        a: 被除数
        b: 除数
    """
    try:
        if b == 0:
            return "错误：除数不能为零"
        result = a / b
        return f"{a} / {b} = {result}"
    except Exception as e:
        return f"计算错误：{e}"
```

### 4. 返回字符串
```python
# ✅ 好：返回字符串
@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    user = {"id": user_id, "name": "张三"}
    return json.dumps(user, ensure_ascii=False)  # 转成 JSON 字符串

# ❌ 不好：返回字典（某些情况可能有问题）
@tool
def get_user_info(user_id: str) -> dict:
    """获取用户信息"""
    return {"id": user_id, "name": "张三"}
```

## 测试工具

每个工具文件都可以直接运行测试：

```python
# 在文件末尾添加
if __name__ == "__main__":
    print("测试工具：")
    print(my_tool.invoke({"param": "test"}))
```

运行测试：
```bash
python tools/weather.py
```

## 项目结构

```
04_custom_tools/
├── main.py                # 6 个示例
├── README.md              # 本文件
└── tools/                 # 工具目录
    ├── weather.py         # 天气工具
    ├── calculator.py      # 计算器工具
    └── web_search.py      # 搜索工具
```

## 运行示例

```bash
# 测试单个工具
python tools/weather.py

# 运行所有示例
python main.py
```

## 下一步

**05_simple_agent** - 使用 `create_agent` 让 AI 自动调用这些工具
