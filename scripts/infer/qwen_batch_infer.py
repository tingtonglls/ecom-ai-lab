import argparse
import json
import re
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
from tqdm import tqdm
import sys
sys.argv = [
	"qwen_batch_infer.py",

	"-m",
	"Qwen/Qwen3-0.6B",

	"-i",
	"dataset/intent_recognition/gold_1500.jsonl",

	"-o",
	"outputs/infer/0.6B_batch_1500.jsonl",

	"-n",
	"1500"
]

MODEL_DEFAULT = "Qwen/Qwen3-0.6B"


def try_parse_json(s: str):
	s = s.strip()
	try:
		return json.loads(s)
	except Exception:
		pass
	start = s.find("{")
	end = s.rfind("}")
	if start != -1 and end != -1 and end > start:
		try:
			return json.loads(s[start:end+1])
		except Exception:
			return None
	return None


def strip_think_tags(s: str) -> str:
	if not s:
		return s
	# remove <think>...</think> blocks
	s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
	# remove any remaining tags
	s = s.replace("<think>", "").replace("</think>", "")
	return s.strip()


def normalize_keywords(k):
	# return list of keywords
	if k is None:
		return []
	if isinstance(k, list):
		return [str(x).strip() for x in k if str(x).strip()]
	if isinstance(k, str):
		# try split by common delimiters
		parts = re.split(r"[,，;；\s]+", k)
		return [p.strip() for p in parts if p.strip()]
	# fallback
	return [str(k)]


def parse_malformed_json(s: str):
	"""Attempt to extract 类型 and 关键词 from malformed JSON-like string."""
	if not s or '{' not in s:
		return None
	# try to find 类型 value
	t = None
	mtype = re.search(r'类型["' + "'" + r']?[:：\"\']*([^\",\}\n]+)', s)
	if mtype:
		t = mtype.group(1).strip().strip('"').strip("'")

	# try to find 关键词 content after 关键词\": or 关键词":
	mk = re.search(r'关键词["' + "'" + r']?[:：\"\']*([^\}\n]+)', s)
	keywords = []
	if mk:
		kraw = mk.group(1)
		# remove trailing quotes/braces
		kraw = kraw.strip().strip('"').strip("'")
		# split by common delimiters
		parts = re.split(r"[,，;；\s]+", kraw)
		keywords = [p.strip().strip('"').strip("'") for p in parts if p.strip()]

	if t is None and not keywords:
		return None
	return {"类型": t or "", "关键词": " ".join(keywords) if keywords else ""}


def infer_file(model_name, prompt_path: Path, input_path: Path, output_path: Path, max_items: int = None):
	print("从ModelScope下载/读取模型...")
	model_dir = snapshot_download(model_name)
	print(f"模型目录：{model_dir}")
	print("加载Tokenizer...")
	global tokenizer, model
	tokenizer = AutoTokenizer.from_pretrained(
	    model_dir,
	    trust_remote_code=True
    )
	print("加载模型... (这可能需要一些时间)")
	dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
	model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True
    )
	

	device = model.device
	print(f"模型加载完成，device={device}")

	template = prompt_path.read_text(encoding="utf-8")

	out_lines = []
	with input_path.open("r", encoding="utf-8") as fin:
		lines = fin.readlines()

	if max_items is not None:
		lines = lines[:max_items]

	for idx, line in enumerate(tqdm(lines, desc="推理进度"), start=1):
		line = line.strip()
		if not line:
			continue
		try:
			item = json.loads(line)
		except Exception:
			continue

		question = item.get("原始问题") or item.get("问题") or item.get("question") or item.get("text") or item.get("问")
		if not question:
			continue

		# run model
		prompt_filled = template.format(question=question)
		messages = [{"role": "user", "content": prompt_filled}]
		text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

		inputs = tokenizer(text, return_tensors="pt")
		inputs = {k: v.to(model.device) for k, v in inputs.items()}

		outputs = model.generate(
			**inputs,
			max_new_tokens=512,
			do_sample=False,
			temperature=0,
			top_p=1.0
		)

		generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]
		raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

		# strip <think> blocks from raw output
		clean_raw = strip_think_tags(raw_response)

		parsed = try_parse_json(clean_raw)
		# if parsing failed, try to recover from common malformed patterns
		if parsed is None:
			parsed = parse_malformed_json(clean_raw)

		# derive predicted label and keywords from parsed JSON when possible
		pred_label = None
		pred_keywords = []
		if isinstance(parsed, dict):
			pred_label = parsed.get("类型") or parsed.get("label") or parsed.get("pred_label")
			k = parsed.get("关键词") or parsed.get("keywords") or parsed.get("pred_keywords")
			pred_keywords = normalize_keywords(k)
		else:
			# fallback heuristics: look for 电商/闲聊 and simple keyword extraction
			if "电商" in clean_raw:
				pred_label = "电商类"
			elif "闲聊" in clean_raw:
				pred_label = "闲聊类"
			# try to find keywords in quotes or after 关键词
			# try to capture keywords more robustly until a closing brace/newline
			m = re.search(r'''关键词["'：:\s]*([^\}\n\r]+)''', clean_raw)
			if m:
				pred_keywords = normalize_keywords(m.group(1))

		# gold fields
		gold_label = item.get("原始分类") or item.get("类型") or item.get("gold_label")
		gold_k = item.get("原始关键词") or item.get("关键词") or item.get("gold_keywords")
		gold_keywords = normalize_keywords(gold_k)

		# put parsed JSON into 模型原始输出 when parsing succeeded, so quotes won't be escaped
		model_output_value = parsed if isinstance(parsed, dict) else clean_raw

		out_obj = {
			"id": idx,
			"原始问题": question,
			"原始分类": gold_label,
			"原始关键词": gold_keywords,
			"模型分类": pred_label,
			"模型提取关键词": pred_keywords,
			"模型原始输出": model_output_value,
			"metadata": {
				"model": model_name,
				"prompt": str(prompt_path),
				"device": str(device),
				"timestamp": datetime.utcnow().isoformat() + "Z"
			}
		}

		out_lines.append(out_obj)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as fout:
		for obj in out_lines:
			fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

	print(f"已写入 {output_path}，共 {len(out_lines)} 条结果")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", "-i", required=True, help="输入 jsonl 文件路径，例如: data/demo_dataset.jsonl")
	parser.add_argument("--output", "-o", default="outputs/batch_infer_results.jsonl", help="输出 jsonl 路径")
	parser.add_argument("--model", "-m", default=MODEL_DEFAULT, help="模型名")
	parser.add_argument("--prompt", "-p", default="prompts/qwen_infer_prompt.txt", help="prompt 模板文件路径")
	parser.add_argument("--max_items", "-n", type=int, default=None, help="最多推理多少条（用于本地小批量测试）")

	try:
		args = parser.parse_args()
	except SystemExit:
		parser.print_help()
		return

	input_path = Path(args.input)
	output_path = Path(args.output)
	prompt_path = Path(args.prompt)
	infer_file(args.model, prompt_path, input_path, output_path, max_items=args.max_items)


if __name__ == "__main__":
	main()

