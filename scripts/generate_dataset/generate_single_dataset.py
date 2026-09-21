import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
load_dotenv()

# 获取 API Key
my_api_key = os.getenv("DEEPSEEK_API_KEY")

#调用deepseek-api 生成数据集
client = OpenAI(
    api_key = my_api_key, 
    base_url = "https://api.deepseek.com"
)

#当前脚本目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
#tools.json路径
tools_json_path = os.path.join(
    script_dir,
    "../configs/tools.json"
)
#dataset-config.json路径
datasetConfig_json_path = os.path.join(
    script_dir,
    "../configs/dataset_config.json"
)
#prompt模版路径
prompt_path = os.path.join(
    script_dir,
    "../prompts/single_intent_prompt.txt"
)
#单意图outputs目录路径
output_dir = os.path.join(
    script_dir,
    "../outputs/single_intent"
)
os.makedirs(output_dir, exist_ok=True)

#读取tools.json
with open(tools_json_path, "r", encoding="utf-8") as f:
    tools = json.load(f)
#读取dataset_config.json
with open(datasetConfig_json_path, "r", encoding="utf-8") as f:
    datasetConfig = json.load(f)
#读取Prompt模版
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()

#每轮次生成数据条数
batch_size = datasetConfig["single_intent"]["batch_size"]

# #选择需要生成数据的目标工具
# target_tools = [
#     "get_user_profile"
# ]


#遍历所有工具
for cur_tool_name, tool_info in tools.items():
    # if cur_tool_name not in target_tools:
    #     continue

    cur_description = tool_info["description"]
    cur_text_augmentation = tool_info["text_augmentation"]
    cur_target_count = datasetConfig["single_intent"]["count_per_tool"]


    #输出文件路径
    output_path = os.path.join(output_dir, f"{cur_tool_name}.jsonl")

    #需要生成轮数(220/22=10轮)
    generate_times = cur_target_count // batch_size  #//是整除

    #ceil()是向上取整，可以用它来确保足够的数量
    # import math
    # generate_times = math.ceil(
    #     cur_target_count / batch_size
    # )

    print(f"\n==============================")  #用f可以直接把加上{}的变量塞进字符串
    print(f"开始生成工具：{cur_tool_name}")
    print(f"目标数量：{cur_target_count}")
    print(f"每轮生成：{batch_size}")
    print(f"共需要生成：{generate_times}轮")
    print(f"==============================\n")

    #拼接prompt
    #format() 字符串格式化:把模板中的{}替换成具体值
    prompt = prompt_template.format(
        tool_name = cur_tool_name,
        description = cur_description,
        text_augmentation = cur_text_augmentation,
        count = batch_size
    ) 
    # print(f"prompt是：\n{prompt}")

    #小批量循环生成
    for i in range(generate_times):
        print(f"正在进行第{i+1}轮生成......")
        

        #调用DeepSeek 
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1.2
        )

        #获取模型输出
        result = response.choices[0].message.content

        #整理输出格式 去除markdown
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()


        #保存 jsonl
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(result + "\n")

    print(f"{cur_tool_name}的数据生成完成")

print("所有工具数据生成完成！")