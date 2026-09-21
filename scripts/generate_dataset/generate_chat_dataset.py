import os
from openai import OpenAI
from dotenv import load_dotenv

# 配置
BATCH_SIZE = 20
GENERATE_TIMES = 35      # 应调用25次API，共生成500条

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 路径
script_dir = os.path.dirname(os.path.abspath(__file__))

prompt_path = os.path.join(
    script_dir,
    "../prompts/chat_prompt.txt"
)

output_path = os.path.join(
    script_dir,
    "../dataset/intent_recognition/chat_intent_500.jsonl"
)

checkpoint_path = os.path.join(
    script_dir,
    "../dataset/intent_recognition/chat_checkpoint.txt"
)

# 读取prompt
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()



# 开始生成
for i in range(GENERATE_TIMES):

    print(f"\n正在生成第 {i+1}/{GENERATE_TIMES} 轮")

    prompt = prompt_template.format(
        count=BATCH_SIZE
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
            temperature=1.0
        )

        result = response.choices[0].message.content
        print("========== 原始返回 ==========")
        print(repr(result))
        print("=============================")

        # 去除 markdown 包装
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

        # 写入 jsonl
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result + "\n")

        print("生成成功")

    except Exception as e:

        print("生成失败：", e)

print("\n全部生成完成")
print(f"输出文件：{output_path}")