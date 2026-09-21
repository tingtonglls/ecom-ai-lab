import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("加载Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("加载模型...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

print(model.device)
print("\n模型加载完成")

question = "手机激活后卡顿能退吗？"

prompt = f"""
/no_think

你是意图识别+关键词提取专家。

判断问题类型：

-电商类:
与商品购买、价格优惠、物流、订单、退款、售后、
退货、换货、评价、库存、支付等相关的问题。

注意：
用户表达可能不直接出现"退款、优惠、售后"等字眼。
只要用户的问题最终需要电商平台、商家或者售后服务来解决，就属于电商类。

-闲聊类
天气、百科、路线、闲聊、常识问答等与电商无关的问题。

如果是电商类，提取关键词。
如果是闲聊类：关键词为空。

要求：
1. 不要输出分析过程
2. 不要输出思考过程
3. 不要输出<think>
4. 直接输出JSON
5. 只能输出JSON

-示例：

问题1：
这个手机有没有优惠

输出1：
{{"类型":"电商类","关键词":"优惠"}}


问题2：
今天天气怎么样

输出2：
{{"类型":"闲聊类","关键词":""}}



问题：

{question}
"""

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("=" * 100)
print(text)
print("=" * 100)

inputs = tokenizer(
    text,
    return_tensors="pt"
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    temperature=0,
    top_p=1.0
)

# 避免prompt被重复输出
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs["input_ids"], outputs)
]

response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print(response)