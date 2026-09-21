import json
import math
import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件
load_dotenv()

# 获取 API Key
my_api_key = os.getenv("DEEPSEEK_API_KEY")

# DeepSeek Client
client = OpenAI(
    api_key=my_api_key,
    base_url="https://api.deepseek.com"
)

# 当前脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# multi_intent_combinations.json路径
combinations_path = os.path.join(
    script_dir,
    "../configs/multi_intent_combinations.json"
)
# dataset_config.json路径
dataset_config_path = os.path.join(
    script_dir,
    "../configs/dataset_config.json"
)
# prompt模板路径
prompt_path = os.path.join(
    script_dir,
    "../prompts/multi_intent_prompt.txt"
)
# tools.json路径
tools_path = os.path.join( 
    script_dir, 
    "../configs/tools.json"
)
# 多意图outputs目录路径
output_dir = os.path.join(
    script_dir,
    "../outputs/multi_intent"
)
os.makedirs(output_dir, exist_ok=True)


# 读取multi_intent_combinations.json
with open(combinations_path, "r", encoding="utf-8") as f:
    combinations = json.load(f)
# 读取dataset_config.json
with open(dataset_config_path, "r", encoding="utf-8") as f:
    dataset_config = json.load(f)
#读取prompt模板
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()
# 读取tools.json 
with open(tools_path, "r", encoding="utf-8") as f: 
    tools_config = json.load(f)


# 获取batch_size
batch_size = dataset_config["multi_intent"]["batch_size"]

target_flows = ["promotion_flow"]
target_tool_chains = [
    # [
    #   "get_current_promotions",
    #   "get_available_coupons"
    # ],
    # [
    #   "get_cart_summary",
    #   "calc_discount"
    # ],
    # [
    #   "get_available_coupons",
    #   "apply_coupon"
    # ],
    # [
    #   "get_cart_summary",
    #   "get_available_coupons",
    #   "calc_discount"
    # ],
    # [
    #   "get_current_promotions",
    #   "calc_discount"
    # ],
    # [
    #   "get_cart_summary",
    #   "get_available_coupons",
    #   "apply_coupon",
    #   "calc_discount"
    # ],
    [
      "get_current_promotions",
      "get_cart_summary",
      "get_available_coupons",
      "apply_coupon",
      "calc_discount"
    ]
]


# 遍历 flow
for flow_name, tool_chains in combinations.items():

    if flow_name not in target_flows:
        continue
    print(f"\n========== 开始生成Flow：{flow_name} ==========\n")

    # 当前flow的总数据量
    flow_total_count = dataset_config["multi_intent"]["flow_count_distribution"][flow_name]
    # 当前flow有多少种组合
    combination_count = len(tool_chains)
    # print(f"\n{flow_name}有{combination_count}种组合。")

    # 每个组合生成数据总量
    count_per_combination = flow_total_count // combination_count
    # 向上取整计算生成轮次数
    generate_times = math.ceil(
        count_per_combination / batch_size
    )

    print(f"当前flow总数据量: {flow_total_count}\n")
    print(f"当前flow有{combination_count}个组合\n")
    print(f"因此每组合应至少生成: {count_per_combination}；")
    print(f"考虑到大模型推理能力，实际每批生成: {batch_size}；")
    print(f"为满足当前组合所需的数据量{count_per_combination}，实际生成轮数为: {generate_times}。")

    #当前Flow的输出文件路径 : 按照6种flow去分别存储多意图问题数据
    output_path = os.path.join(
        output_dir,
        f"{flow_name}.jsonl"
    )

    #遍历该轮flow下的所有工具组合
    for tool_chain in tool_chains:

        if tool_chain not in target_tool_chains:
            continue
        print(f"\n当前工具链:{tool_chain}")

        # Prompt中字段tool_context的动态拼接        
        tool_context = ""
        for tool_name in tool_chain:
            tool_context += (
                f"\n工具名：{tool_name}\n"
                f"功能描述：{tools_config[tool_name]['description']}\n"
                f"用户可能这样表达：\n"
            )
            for text in tools_config[tool_name]["text_augmentation"]:
                tool_context += f"- {text}\n"

        # Prompt填充
        prompt = prompt_template.format(
            count=batch_size,
            tool_chain=tool_chain,
            tool_context=tool_context
        )

        # print(f"\n当前prompt为：{prompt}")

        # 小批量循环生成
        for i in range(generate_times):
            print(f"\n正在生成第 {i+1}/{generate_times} 批...")

            try:
                
                #调用deepseek
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt                                               }
                    ],
                    temperature=1.2
                )

                result = response.choices[0].message.content
                print("========== 原始返回 ==========")
                print(repr(result))
                print("=============================")
                #整理输出格式
                result = result.replace("```json", "")
                result = result.replace("```", "")
                result = result.strip()

                #写入jsonl
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(result + "\n")

                print("本批生成成功")
                line_count = len(
                    [
                        line
                        for line in result.split("\n")
                        if line.strip()
                    ]
                )
                print(f"本批生成 {line_count} 条")

            except Exception as e:
                print("生成失败")
                print(e)

        print(f"当前工具组合{tool_chain}生成完成")        

    print(f"\nFlow {flow_name} 全部生成完成") 

print("\n===========所有多意图数据生成完成=============")           