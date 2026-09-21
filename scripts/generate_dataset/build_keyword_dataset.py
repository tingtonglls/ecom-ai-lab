import os
import json
from openai import OpenAI
from dotenv import load_dotenv

###该脚本搭配旧版本e-commerce——prompt，只能实现调用一次api只生成一条数据、对应处理一个用户问题，故不再使用。
###但因其流程清晰，故保留以便复习


BATCH_SIZE = 5

# 要求数据输出在dataset/intent_recognition/ecommerce_intent_1000.jsonl

# 加载 .env文件
load_dotenv()

# 获取 API Key
my_api_key = os.getenv("DEEPSEEK_API_KEY")

# 调用deepseek-api 生成数据集
client = OpenAI(
    api_key = my_api_key,
    base_url = "https://api.deepseek.com"
)

# 当前脚本目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 输出目录
intentRecognition_dir = os.path.join(
    script_dir,
    "../dataset/intent_recognition"
)
# 输出文件路径
ecommerceIntent_path = os.path.join(
    intentRecognition_dir,
    "ecommerce_intent_1000.jsonl"
)

os.makedirs(intentRecognition_dir, exist_ok=True)
# 输入文件intent_test.jsonl路径
intentTest_jsonl_path = os.path.join(
    script_dir,
    "../dataset/tool_calling_trace/test/intent_test.jsonl"
)
# 模版prompt路径
prompt_path = os.path.join(
    script_dir,
    "../prompts/e-commerce_prompt.txt"
)

# checkpoint文件
checkpoint_path = os.path.join(
    intentRecognition_dir,
    "checkpoint.txt"
)


# 读取intent_test.jsonl
samples = []
with open(intentTest_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip() #移除字符串首尾空格
        if not line:
            continue
        
        try:
            samples.append(json.loads(line))
        except:
            continue
    
print(f"读取测试集成功，共{len(samples)}条")

# 读取prompt模板
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# 读取checkpoint
if os.path.exists(checkpoint_path):
    with open(checkpoint_path,"r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            start_index = int(content)
        else:
            start_index = 0
else:
    start_index = 0

print(f"本次运行从第{start_index + 1}条开始处理")

# 第一次运行时创建输出文件
if start_index == 0:
    with open(ecommerceIntent_path, "w", encoding="utf-8"):
        pass

# 开始批量生成
success_count = start_index

end_index = start_index + BATCH_SIZE

for idx, sample in enumerate(samples[start_index:end_index], start=start_index + 1):

    question = sample["问题"]
    print(f"用户问题是：{question}\n")
    prompt = prompt_template.format(
        question = question
    )

    try:
        print(f"\n[{idx}/{len(samples)}]:{question}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                { 
                    "role": "user", 
                    "content": prompt 
                }
            ],
            temperature=0.3
        )
        result = response.choices[0].message.content

        print("========== 原始返回 ==========") 
        print(repr(result)) 
        print("=============================") 
        # 清理markdown 
        result = result.replace("```json", "") 
        result = result.replace("```", "") 
        result = result.strip()

        # json校验
        json_obj = json.loads(result)

        # 写入jsonl
        with open(ecommerceIntent_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(json_obj, ensure_ascii=False) + "\n"
            )
        
        # 更新checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            f.write(str(idx))
        
        print(f"第{idx}条成功。")
        success_count += 1
    
    except Exception as e:
        print("生成失败：", e)

print("\n======================")
print(f"完成，本次新增至第{success_count} 条")
print(f"输出文件：{ecommerceIntent_path}")
print("======================")