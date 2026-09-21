import argparse
import json
from pathlib import Path
import sys
sys.argv = ["classification.py", "-i", "outputs/infer/1.7B_batch_1500.jsonl"]

def compute_metrics(tp, fp, fn, tn):
	total = tp + fp + fn + tn
	accuracy = (tp + tn) / total if total else 0.0
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	return {
		'total': total,
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'f1': f1,
	}


def evaluate_classification(input_path: Path, gold_field: str, pred_field: str, pos_label: str, neg_label: str):
	tp = fp = fn = tn = 0
	total = 0
	per_examples = []
	with input_path.open('r', encoding='utf-8') as fin:
		for line in fin:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			gold = obj.get(gold_field) or obj.get('gold_label') or obj.get('原始分类')
			pred = obj.get(pred_field) or obj.get('pred_label') or obj.get('模型分类')
			if gold is None or pred is None:
				continue
			gold = str(gold).strip()
			pred = str(pred).strip()
			total += 1

			# determine sample kind and counts
			sample_kind = None
			if gold == pos_label:
				if pred == pos_label:
					tp += 1
					sample_kind = 'TP'
				else:
					fn += 1
					sample_kind = 'FN'
			elif gold == neg_label:
				if pred == pos_label:
					fp += 1
					sample_kind = 'FP'
				else:
					tn += 1
					sample_kind = 'TN'
			else:
				# treat unknown gold as neg unless equals pos
				if pred == pos_label:
					fp += 1
					sample_kind = 'FP'
				else:
					tn += 1
					sample_kind = 'TN'

			per_examples.append({
				'id': obj.get('id'),
				'问题': obj.get('原始问题') or obj.get('question') or obj.get('问题'),
				'真实分类': gold,
				'预测分类': pred,
				'分类结果': (gold == pred),
				'样本种类': sample_kind,
			})

	metrics = compute_metrics(tp, fp, fn, tn)
	return tp, fp, fn, tn, metrics, per_examples


def print_confusion(tp, fp, fn, tn, pos_label='电商类', neg_label='闲聊类'):
	print('Confusion matrix:')
	print(f'\t\tPred={pos_label}\tPred={neg_label}')
	print(f'Gold={pos_label}\tTP={tp}\tFN={fn}')
	print(f'Gold={neg_label}\tFP={fp}\tTN={tn}')


def main():
	parser = argparse.ArgumentParser(description='Evaluate classification metrics for intent results')
	parser.add_argument('--input', '-i', required=True, help='输入 jsonl 文件路径（每行 JSON）')
	parser.add_argument('--gold-field', default='原始分类', help='参考分类字段名，默认 原始分类')
	parser.add_argument('--pred-field', default='模型分类', help='预测分类字段名，默认 模型分类')
	parser.add_argument('--pos-label', default='电商类', help='正类标签，默认 电商类')
	parser.add_argument('--neg-label', default='闲聊类', help='负类标签，默认 闲聊类')

	args = parser.parse_args()
	input_path = Path(args.input)
	tp, fp, fn, tn, metrics, per_examples = evaluate_classification(input_path, args.gold_field, args.pred_field, args.pos_label, args.neg_label)

	print('\nClassification metrics:')
	print(f"Total: {metrics['total']}")
	print(f"Accuracy: {metrics['accuracy']:.4f}")
	print(f"Precision: {metrics['precision']:.4f}")
	print(f"Recall: {metrics['recall']:.4f}")
	print(f"F1: {metrics['f1']:.4f}\n")

	print_confusion(tp, fp, fn, tn, pos_label=args.pos_label, neg_label=args.neg_label)

	# write per-example jsonl
	out_dir = Path('outputs/eval')
	out_dir.mkdir(parents=True, exist_ok=True)
	per_file = out_dir / '1.7B_classification_per_result.jsonl'
	with per_file.open('w', encoding='utf-8') as fout:
		for ex in per_examples:
			fout.write(json.dumps(ex, ensure_ascii=False) + '\n')

	# write summary csv
	import csv
	summary_csv = out_dir / '1.7B_classification_summary.csv'
	with summary_csv.open('w', encoding='utf-8', newline='') as cf:
		writer = csv.writer(cf)
		writer.writerow(['metric', 'value'])
		writer.writerow(['total', metrics['total']])
		writer.writerow(['accuracy', metrics['accuracy']])
		writer.writerow(['precision', metrics['precision']])
		writer.writerow(['recall', metrics['recall']])
		writer.writerow(['f1', metrics['f1']])
		writer.writerow(['TP', tp])
		writer.writerow(['FP', fp])
		writer.writerow(['FN', fn])
		writer.writerow(['TN', tn])

	print(f"\nWrote per-example results to {per_file}")
	print(f"Wrote summary CSV to {summary_csv}")


if __name__ == '__main__':
	main()

