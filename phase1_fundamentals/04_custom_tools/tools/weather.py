"""
 自定义工具:天气查询
 =====================

 使用@tool装饰器创建工具(Langchain 1.0 推荐方式)
"""

from langchain_core.tools import tool

@tool
def get_weather(city: str)-> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京","上海"

    返回:
        天气信息字符串
    """
    # 模拟天气数据(实际应用中调用真实API)
    weather_data={
        "北京": "晴天，温度25度",
        "上海": "阴天，温度23度",
        "广州": "雨天，温度18度",
        "深圳": "雷阵雨，温度20度"
    }
    return weather_data.get(city,f"未找到{city}的天气信息")


# 测试
if __name__ == "__main__":
    print(get_weather.invoke({"city":"北京"}))