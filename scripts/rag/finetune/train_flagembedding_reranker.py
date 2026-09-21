import csv
import os
import shutil
import tempfile
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


DEFAULT_BASE_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TRAIN_DATA = Path(
    "dataset/tool_calling_trace/train/intent_train_flagembedding_reranker.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("model/my-bge-reranker")
DEFAULT_LOG_DIR = Path("outputs/rag/finetune/bge-reranker")
DEFAULT_CONFIG_PATH = Path("configs/rag-finetune/bge-reranker_finetune_config.yaml")
LOSS_FILENAMES = {"training_loss.csv", "training_loss.png"}


def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()
    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "dataset").exists():
            return path
    raise FileNotFoundError("无法定位项目根目录，请确认脚本位于仓库内。")


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def format_lr(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("-0", "-").replace("+0", "")


def build_experiment_name(batch_size: int, num_epochs: int, learning_rate: float) -> str:
    return f"bs{batch_size}_ep{num_epochs}_lr{format_lr(learning_rate)}"


def count_training_rows(train_data_path: Path) -> int:
    if not train_data_path.exists():
        return 0
    with train_data_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def clean_experiment_loss_dir(log_base_dir: Path) -> None:
    log_base_dir.mkdir(parents=True, exist_ok=True)
    for path in log_base_dir.iterdir():
        if path.name in LOSS_FILENAMES:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def append_history_csv(csv_path: Path, metrics_row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "model",
        "base_model",
        "dataset_rows",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "lr",
        "epochs",
        "train_group_size",
        "query_max_len",
        "passage_max_len",
        "global_step",
        "runtime_seconds",
        "samples_per_second",
        "steps_per_second",
        "final_loss",
        "final_learning_rate",
        "seed",
        "output_dir",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics_row)


def get_last_learning_rate(trainer: Any) -> Optional[float]:
    scheduler = getattr(trainer, "lr_scheduler", None)
    if scheduler is None:
        return None
    try:
        last_lrs = scheduler.get_last_lr()
    except Exception:
        return None
    if not last_lrs:
        return None
    try:
        return float(last_lrs[0])
    except (TypeError, ValueError):
        return None


def extract_loss_series(log_history: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for index, entry in enumerate(log_history, start=1):
        if not isinstance(entry, dict) or not isinstance(entry.get("loss"), (int, float)):
            continue
        step = entry.get("step", index)
        epoch = entry.get("epoch", "")
        row = {
            "step": float(step) if isinstance(step, (int, float)) else float(index),
            "loss": float(entry["loss"]),
        }
        if isinstance(epoch, (int, float)):
            row["epoch"] = float(epoch)
        rows.append(row)
    return rows


def save_loss_outputs(loss_rows: List[Dict[str, float]], loss_path: Path, plot_path: Path) -> None:
    if not loss_rows:
        return

    with loss_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "loss"])
        for row in loss_rows:
            writer.writerow([int(row["step"]), row.get("epoch", ""), row["loss"]])

    steps = [int(row["step"]) for row in loss_rows]
    losses = [row["loss"] for row in loss_rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, marker="o", linewidth=1.2)
    plt.title("bge-reranker training loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def patch_tokenizer_prepare_for_model() -> None:
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return

    if hasattr(PreTrainedTokenizerBase, "prepare_for_model"):
        return

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        truncation=None,
        max_length=None,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=True,
        **kwargs,
    ):
        first_ids = list(ids)
        second_ids = [] if pair_ids is None else list(pair_ids)
        cls_id = getattr(self, "cls_token_id", None)
        sep_id = getattr(self, "sep_token_id", None)
        pad_id = getattr(self, "pad_token_id", 0)
        pad_token_type_id = getattr(self, "pad_token_type_id", 0)
        class_name = self.__class__.__name__.lower()
        use_roberta_pair = bool(second_ids) and "roberta" in class_name

        if max_length is not None:
            special_count = 0
            if cls_id is not None:
                special_count += 1
            if sep_id is not None:
                special_count += 2 if use_roberta_pair else 1
                if second_ids:
                    special_count += 1

            overflow = len(first_ids) + len(second_ids) + special_count - max_length
            if overflow > 0:
                if truncation == "only_first":
                    first_ids = first_ids[: max(0, len(first_ids) - overflow)]
                else:
                    second_ids = second_ids[: max(0, len(second_ids) - overflow)]

        if cls_id is None or sep_id is None:
            input_ids = first_ids + second_ids
        elif use_roberta_pair:
            input_ids = [cls_id] + first_ids + [sep_id, sep_id] + second_ids + [sep_id]
        elif second_ids:
            input_ids = [cls_id] + first_ids + [sep_id] + second_ids + [sep_id]
        else:
            input_ids = [cls_id] + first_ids + [sep_id]

        if padding in ("max_length", True) and max_length is not None and len(input_ids) < max_length:
            input_ids = input_ids + [pad_id] * (max_length - len(input_ids))

        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = [0 if token_id == pad_id else 1 for token_id in input_ids]
        if return_token_type_ids:
            output["token_type_ids"] = [pad_token_type_id] * len(input_ids)
        return output

    PreTrainedTokenizerBase.prepare_for_model = prepare_for_model


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到 reranker 配置文件：{config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"reranker 配置必须是 YAML 对象：{config_path}")
    return config


def get_section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"配置段 {name!r} 必须是 YAML 对象")
    return section


def build_args(config: Dict[str, Any]) -> Namespace:
    model_config = get_section(config, "model")
    dataset_config = get_section(config, "dataset")
    training_config = get_section(config, "training")
    reranker_config = get_section(config, "reranker")
    logging_config = get_section(config, "logging")

    return Namespace(
        model_name_or_path=model_config.get("name", DEFAULT_BASE_MODEL),
        train_data=Path(dataset_config.get("train_data", DEFAULT_TRAIN_DATA)),
        output_dir=Path(model_config.get("output_dir", DEFAULT_OUTPUT_DIR)),
        log_dir=Path(logging_config.get("log_dir", DEFAULT_LOG_DIR)),
        epochs=int(training_config.get("epochs", 2)),
        batch_size=int(training_config.get("batch_size", 16)),
        learning_rate=float(training_config.get("learning_rate", 1e-5)),
        gradient_accumulation_steps=int(
            training_config.get("gradient_accumulation_steps", 1)
        ),
        train_group_size=int(reranker_config.get("train_group_size", 8)),
        query_max_len=int(reranker_config.get("query_max_len", 128)),
        passage_max_len=int(reranker_config.get("passage_max_len", 256)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.1)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        logging_steps=int(logging_config.get("logging_steps", 20)),
        save_total_limit=int(training_config.get("save_total_limit", 3)),
        seed=int(training_config.get("seed", 42)),
        fp16=bool(training_config.get("fp16", False)),
        bf16=bool(training_config.get("bf16", False)),
        gradient_checkpointing=bool(
            training_config.get("gradient_checkpointing", False)
        ),
    )


def main() -> None:
    root = find_project_root()
    config = load_config(resolve_path(DEFAULT_CONFIG_PATH, root))
    args = build_args(config)

    train_data = resolve_path(args.train_data, root)
    base_output_dir = resolve_path(args.output_dir, root)
    base_log_dir = resolve_path(args.log_dir, root)
    experiment_name = build_experiment_name(
        args.batch_size,
        args.epochs,
        args.learning_rate,
    )
    model_output_dir = base_output_dir / experiment_name
    log_base_dir = base_log_dir / experiment_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    clean_experiment_loss_dir(log_base_dir)

    loss_path = log_base_dir / "training_loss.csv"
    loss_plot_path = log_base_dir / "training_loss.png"
    experiment_history_path = base_log_dir / "experiment_history.csv"
    cache_path = Path(tempfile.gettempdir()) / "flagembedding_reranker_cache" / experiment_name
    trainer_logging_dir = Path(tempfile.gettempdir()) / "flagembedding_reranker_logs" / experiment_name

    try:
        from FlagEmbedding.abc.finetune.reranker import (
            AbsRerankerDataArguments,
            AbsRerankerModelArguments,
            AbsRerankerTrainingArguments,
        )
        import FlagEmbedding.finetune.reranker.encoder_only.base.runner as reranker_runner
        from FlagEmbedding.finetune.reranker.encoder_only.base import (
            EncoderOnlyRerankerRunner,
        )
        from FlagEmbedding.finetune.reranker.encoder_only.base.trainer import (
            EncoderOnlyRerankerTrainer,
        )
    except ImportError as exc:
        raise SystemExit(
            "当前环境未安装可用的 FlagEmbedding reranker 微调接口。"
            "请先激活 ecommerce-agent 环境，并安装/升级：pip install -U 'FlagEmbedding[finetune]'"
        ) from exc

    if not train_data.exists():
        raise FileNotFoundError(f"找不到训练数据：{train_data}")

    print("开始微调模型：bge-reranker")
    print(f"基础模型：{args.model_name_or_path}")
    print(f"训练数据：{train_data}")
    print(f"训练样本数：{count_training_rows(train_data)}")
    print(f"实验名称：{experiment_name}")
    print(f"模型保存目录：{model_output_dir}")
    print(f"Loss 与实验汇总目录：{log_base_dir}")
    print(
        "训练参数："
        f"batch_size={args.batch_size}, "
        f"grad_acc={args.gradient_accumulation_steps}, "
        f"epochs={args.epochs}, "
        f"lr={args.learning_rate}, "
        f"train_group_size={args.train_group_size}"
    )

    model_args = AbsRerankerModelArguments(
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=False,
        model_type="encoder",
    )
    data_args = AbsRerankerDataArguments(
        train_data=[str(train_data)],
        cache_path=str(cache_path),
        train_group_size=args.train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        pad_to_multiple_of=8,
        knowledge_distillation=False,
        shuffle_ratio=0.0,
    )
    training_args = AbsRerankerTrainingArguments(
        output_dir=str(model_output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_steps=1000,
        save_total_limit=args.save_total_limit,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        do_train=True,
        do_eval=False,
        dataloader_drop_last=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=str(trainer_logging_dir),
        seed=args.seed,
    )
    patch_tokenizer_prepare_for_model()

    class CompatibleEncoderOnlyRerankerTrainer(EncoderOnlyRerankerTrainer):
        def __init__(self, *args, tokenizer=None, processing_class=None, **kwargs):
            if processing_class is None:
                processing_class = tokenizer
            self.tokenizer = processing_class
            super().__init__(*args, processing_class=processing_class, **kwargs)
            self.tokenizer = processing_class

    reranker_runner.EncoderOnlyRerankerTrainer = CompatibleEncoderOnlyRerankerTrainer

    start_time = time.time()
    runner = EncoderOnlyRerankerRunner(model_args, data_args, training_args)
    train_output = runner.trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    runner.trainer.save_model()
    elapsed_seconds = int(time.time() - start_time)

    training_metrics = getattr(train_output, "metrics", {}) or {}
    if not isinstance(training_metrics, dict):
        training_metrics = dict(training_metrics)

    loss_rows = extract_loss_series(getattr(runner.trainer.state, "log_history", []))
    final_loss = training_metrics.get("train_loss")
    if final_loss is None and loss_rows:
        final_loss = loss_rows[-1]["loss"]
    if final_loss is not None:
        try:
            final_loss = float(final_loss)
        except (TypeError, ValueError):
            final_loss = None

    final_learning_rate = get_last_learning_rate(runner.trainer)
    if final_learning_rate is None:
        final_learning_rate = training_metrics.get("learning_rate")

    experiment_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "bge-reranker",
        "base_model": args.model_name_or_path,
        "dataset_rows": count_training_rows(train_data),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "train_group_size": args.train_group_size,
        "query_max_len": args.query_max_len,
        "passage_max_len": args.passage_max_len,
        "global_step": int(getattr(train_output, "global_step", 0) or 0),
        "runtime_seconds": training_metrics.get("train_runtime", elapsed_seconds),
        "samples_per_second": training_metrics.get("train_samples_per_second"),
        "steps_per_second": training_metrics.get("train_steps_per_second"),
        "final_loss": "" if final_loss is None else f"{final_loss:.6f}",
        "final_learning_rate": ""
        if final_learning_rate is None
        else f"{float(final_learning_rate):.6e}",
        "seed": args.seed,
        "output_dir": str(model_output_dir.relative_to(root)),
    }
    append_history_csv(experiment_history_path, experiment_row)
    save_loss_outputs(loss_rows, loss_path, loss_plot_path)

    print(f"微调完成，模型已保存到：{model_output_dir}")
    print(f"实验汇总已保存到：{experiment_history_path}")
    if loss_rows:
        print(f"训练 loss 曲线已保存到：{loss_path}")
        print(f"训练 loss 图片已保存到：{loss_plot_path}")
    else:
        print("未捕获到 loss 日志，跳过 loss CSV 和曲线图保存。")


if __name__ == "__main__":
    main()
