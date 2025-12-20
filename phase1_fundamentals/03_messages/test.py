"""
测试脚本 - 验证对话历史管理
"""
import os
from  dotenv import load_dotenv
from langchain.chat_models import init_chat_model\

load_dotenv()

KEY=os.getenv("DEEPSEEK_API_KEY")

model=init_chat_model(
    model="deepseek:deepseek-chat",
    api_key=KEY
)

def test_conversation_memory():
    """测试AI 是否记住对话"""
    print("\n 测试: AI 对话记忆")

    conversation= [
        {"role": "system", "content": "你是一个助手"}
    ]

    # 告诉AI 用户名字
    conversation.append({"role": "user", "content": "我叫张三"})

    r1=model.invoke(conversation)
    print(f"用户: 我叫张三")
    print(f"AI: {r1.content}")

    # 测试记忆
    conversation.append({"role": "assistant", "content": r1.content})

    conversation.append({"role": "user", "content": "我叫什么名字？"})

    r2=model.invoke(conversation)
    print(f"\n用户: 我叫什么名字？")
    print(f"AI: {r2.content}")
    if "张三" in r2.content :
        print("\n✅ AI 记住了用户名")
        return  True
    else:
        print("\n❌ AI 不记得你叫张三！")
        return False


def test_optimize_history():
    """测试历史优化函数"""
    print("\n 测试: 对话历史优化")
    print("="*50)

    def keep_recent_messages(messages,max_pairs=3):
        """
        保留最近的N轮对话
        参数：
            03_messages: 完整消息列表
            max_pairs: 保留的对话轮数
        返回:
            优化后的消息列表
        """
        # 分离system消息和对话消息
        system_msgs=[m for m in messages if m.get("role")=="system"]
        conversation_msgs=[ m for m in messages if m.get("role") !="system"]
        recent=conversation_msgs[-(max_pairs*2):]

        return system_msgs+recent

    # 创建长历史
    long_conversation = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"},
        {"role": "user", "content": "问题3"},
        {"role": "assistant", "content": "回答3"},
        {"role": "user", "content": "问题4"},
        {"role": "assistant", "content": "回答4"},
    ]
    print(f"原始消息数:{len(long_conversation)}")

    # 优化
    optimized=keep_recent_messages(long_conversation,max_pairs=2)
    print(f"优化后消息数:{len(optimized)}")

    # 验证
    expected=1+(2*2)
    if len(optimized)==expected:
        print("\n✅ 历史优化成功")
        return True
    else:
        print("\n❌ 历史优化失败")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 运行测试")
    print("="*50)

    results=[]
    results.append(test_conversation_memory())
    results.append(test_optimize_history())

    print("\n" + "=" * 50)
    print(" 测试结果")
    print("=" * 50)
    print(f"通过: {sum(results)}/{len(results)}")

    if all(results):
        print("\n 测试通过")
    else:
        print("\n 测试失败")
