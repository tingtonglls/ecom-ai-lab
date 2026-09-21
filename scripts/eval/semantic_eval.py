import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import sys
sys.argv = [
    "semantic_eval.py",

        "--input",
        "outputs/infer/1.7B_batch_1500.jsonl",

        "--prompt",
        "prompts/semantic_eval_prompt.txt",

        "--outdir",
        "outputs/eval",

        "--model",
        "deepseek-chat"
]

# 工具函数

def normalize_keywords(value):
    """把关键词统一转成字符串，方便放进 prompt。"""
    if value is None:
        return ""

    if isinstance(value, list):
        return "，".join([str(x).strip() for x in value if str(x).strip()])

    return str(value).strip()


def parse_ds_json(text):
    """解析 deepseek 返回的 JSON。"""
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1)

    if text.startswith("```"):
        text = text.replace("```", "", 1)

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()
    return json.loads(text)


def normalize_binary(value, default=0):
    """把模型返回的二分类结果统一转成 0/1。"""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value >= 1 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "是", "对", "t"}:
            return 1
    return default


def call_deepseek(client, model_name, prompt):
    """调用 deepseek 做语义评测。"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {# 评测任务，要约束角色比如system
                "role": "system",
                "content": "你是一个严格的关键词语义评测器，只输出JSON。"
            },
            {# 生成任务
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content



# 主程序

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Qwen推理结果jsonl文件路径"
    )

    parser.add_argument(
        "--prompt",
        default="prompts/semantic_eval_prompt.txt",
        help="语义评测prompt路径"
    )

    parser.add_argument(
        "--outdir",
        default="outputs/eval",
        help="评测输出目录"
    )

    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="DeepSeek模型名"
    )

    args = parser.parse_args()

    # 读取 .env
    load_dotenv()

    my_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not my_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY，请先在 .env 文件中配置。")

    client = OpenAI(
    api_key = my_api_key, 
    base_url = "https://api.deepseek.com"
    )

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    per_score_path = outdir / "1.7B_semantic_per_score.jsonl"
    summary_path = outdir / "1.7B_semantic_summary.csv"

    # 读取 prompt 模板
    with prompt_path.open("r", encoding="utf-8") as f:
        prompt_template = f.read()

    total = 0
    strict_match_count = 0
    loose_match_count = 0
    evaluated_ids = set()
    input_total = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            per_score_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()

            if not line:
                continue

            # count total input lines (non-empty)
            input_total += 1

            item = json.loads(line)

            gold_label = (
                item.get("真实分类")
                or item.get("原始分类")
                or item.get("gold_label")
                or ""
            )

            # 只在真实分类和模型预测都为电商类（即样本种类为 TP）时才评测关键词语义
            pred_label = (
                item.get("预测分类")
                or item.get("模型分类")
                or item.get("pred_class")
                or item.get("predicted_class")
                or item.get("模型预测")
                or ""
            )

            if gold_label != "电商类" or pred_label != "电商类":
                continue

            question = (
                item.get("问题")
                or item.get("原始问题")
                or item.get("question")
                or ""
            )

            gold_keywords = normalize_keywords(
                item.get("真实关键词")
                or item.get("原始关键词")
                or item.get("gold_keywords")
            )

            pred_keywords = normalize_keywords(
                item.get("预测关键词")
                or item.get("模型提取关键词")
                or item.get("pred_keywords")
            )

            prompt = prompt_template.format(
                question=question,
                gold=gold_keywords,
                pred=pred_keywords
            )

            try:
                result_text = call_deepseek(
                    client=client,
                    model_name=args.model,
                    prompt=prompt
                )

                result_json = parse_ds_json(result_text)

                strict_match = normalize_binary(result_json.get("strict_match", 0))
                loose_match = normalize_binary(result_json.get("loose_match", 0))
                reason = result_json.get("reason", "")

            except Exception as e:
                strict_match = 0
                loose_match = 0
                reason = f"DeepSeek评测失败：{e}"

            total += 1
            strict_match_count += strict_match
            loose_match_count += loose_match

            out_item = {
                "id": item.get("id"),
                "问题": question,
                "真实关键词": gold_keywords,
                "预测关键词": pred_keywords,
                "严格匹配": strict_match,
                "宽松匹配": loose_match,
                "原因": reason
            }

            fout.write(
                json.dumps(out_item, ensure_ascii=False)
                + "\n"
            )

            # record evaluated id for later cross-check with classification results
            try:
                evaluated_ids.add(item.get('id'))
            except Exception:
                pass

            print(
                f"已评测 {total} 条，"
                f"strict={strict_match}, loose={loose_match}"
            )

    strict_match_rate = round(strict_match_count / total, 4) if total else 0.0
    loose_match_rate = round(loose_match_count / total, 4) if total else 0.0

    # compute classification true positives for evaluated ids (if classification results exist)
    classification_tp = 0
    class_file = outdir / '1.7B_classification_per_result.jsonl'
    if class_file.exists():
        try:
            with class_file.open('r', encoding='utf-8') as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cobj = json.loads(line)
                    except Exception:
                        continue
                    cid = cobj.get('id')
                    kind = cobj.get('样本种类') or cobj.get('sample_kind') or cobj.get('样本类型')
                    if cid in evaluated_ids and kind == 'TP':
                        classification_tp += 1
        except Exception:
            classification_tp = 0

    semantic_eval_count = total

    # write summary in requested order and without duplicate average
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["total", input_total])
        writer.writerow(["classification_tp", classification_tp])
        writer.writerow(["semantic_eval_count", semantic_eval_count])
        writer.writerow(["strict_match_count", strict_match_count])
        writer.writerow(["strict_match_rate", strict_match_rate])
        writer.writerow(["loose_match_count", loose_match_count])
        writer.writerow(["loose_match_rate", loose_match_rate])

    print("\n语义评测完成")
    print(f"逐条结果：{per_score_path}")
    print(f"汇总结果：{summary_path}")


if __name__ == "__main__":
    main()