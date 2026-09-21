import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 一次API处理多少条问题
BATCH_SIZE = 10

# 一次程序运行最多调用多少次API
MAX_BATCHES = 2

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
success_count = 0

batch_count = 0

while (start_index < len(samples) and batch_count < MAX_BATCHES):

    end_index = min(
        start_index + BATCH_SIZE,
        len(samples)
    )

    batch_samples = samples[start_index:end_index]

    print(f"\n处理样本：{start_index + 1} ~ {end_index}")

    # 拼接问题列表
    questions_text = ""
    for i, sample in enumerate(batch_samples, start=1):
        question = sample["问题"]
        questions_text += (f"{i}. {question}\n")

    # 动态拼接Prompt

    prompt = prompt_template.format(
        questions=questions_text
    )

    try:

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

        # print("========== 原始返回 ==========")
        # print(repr(result[:1000]))
        # print("=============================")

        # 清理markdown

        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        # 逐行解析jsonl
        lines = result.splitlines()

        valid_count = 0

        with open(ecommerceIntent_path, "a", encoding="utf-8") as f:

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:

                    json_obj = json.loads(line)

                    f.write(json.dumps(json_obj, ensure_ascii=False)+ "\n")

                    valid_count += 1

                except Exception:

                    print("JSON解析失败：", line)

        print(f"本批成功写入 {valid_count} 条")

        success_count += valid_count

        # 更新checkpoint
        start_index = end_index
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            f.write(str(start_index))

        batch_count += 1

    except Exception as e:

        print("API调用失败：", e)

        break


# 结束统计

print("\n======================")
print(f"本次API调用次数：{batch_count}")
print(f"本次新增数据：{success_count}条")
print(f"当前checkpoint：{start_index}")
print(f"输出文件：{ecommerceIntent_path}")
print("======================")