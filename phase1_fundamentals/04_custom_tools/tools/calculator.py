"""
自定义工具： 计算器

演示带多个参数的工具
"""

from langchain_core.tools import tool

@tool
def calculator(operation: str,a:float,b:float)->str:
    """
    计算器工具，用于计算两个数字之间的运算

    参数:
        operation: 运算符，如"+","-","*","/"
        a: 数字1
        b: 数字2

    返回:
        运算结果字符串
    """
    operations={
        "add":lambda x,y:x+y,
        "sub":lambda x,y:x-y,
        "mul":lambda x,y:x*y,
        "div":lambda x,y:x/y if y!=0 else "除数不能为0"
    }

    if operation not in operations:
        return "无效的运算符"

    try:
        result = operations[operation](a, b)
        return f"{a} {operation} {b} = {result}"
    except Exception as e:
        return str(e)

#测试工具
if __name__ == "__main__":
    print(calculator.invoke({"operation":"add","a":1,"b":2}))
    print(calculator.invoke({"operation":"sub","a":5,"b":2}))
    print(calculator.invoke({"operation":"mul","a":3,"b":4}))
    print(calculator.invoke({"operation":"div","a":10,"b":2}))