# 提示词模板示例库

这个目录包含各种实用的提示词模板示例。

## 文件说明

### template_library.py

包含完整的可复用模板库，分类如下:

- **翻译类模板**
  - TREANSLATOR -专业翻译

- **编程类模板**
  - CODE-GENERATOR -代码生成
  - CODE-REVIEWER -代码审核
  - CODE-EXPLAINER -代码解释
  - CODE-HELPER -调式助手
  
- **内容创作类模板**
  - SUMMARIZER -内容摘要
  - EMAIL-WRITER -邮件编写
  - ARTICLE-WRITER -文章编写

- **教育类模板**
  - TUTOR - 教学辅导
  - QUIZ_GENERATOR - 测验生成

- **商务类模板**
  - PRODUCT_DESCRIPTION - 产品描述
  - MARKET_ANALYSIS - 市场分析

- **客户服务类模板**
  - CUSTOMER_SUPPORT - 客户服务
  - FAQ_RESPONDER - FAQ回答

- **数据分析类模板**
  - DATA_ANALYZER - 数据分析
  - REPORT_GENERATOR - 报告生成

## 使用方法

```python
from examples.template_library import TemplateLibrary
#使用翻译模板
message=TemplateLibrary.TRANSLATOR.format_messages(
    source_lang="英文",
    target_lang="中文",
    text="你好"
)
response=model.invoke(message)
print(response.content)
```

## 测试模板库
```python
# 创建英译中专用模板
from examples.template_library import TemplateLibrary

en_to_zh = TemplateLibrary.TRANSLATOR.partial(
    source_lang="英语",
    target_lang="中文"
)

# 使用
messages = en_to_zh.format_messages(text="Hello")
```