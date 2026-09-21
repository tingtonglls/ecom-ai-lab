import csv
import io
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


MODEL_MAP: Dict[str, Dict[str, str]] = {
    "bge-base": {
        "base_model": "BAAI/bge-base-zh-v1.5",
        "default_output": "model/my-bge-base",
    },
    "bge-large": {
        "base_model": "BAAI/bge-large-zh-v1.5",
        "default_output": "model/my-bge-large",
    },
}

CONFIG_DIR = Path("configs/rag-finetune")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "bge_finetune_config.yaml"


@contextmanager
def capture_and_tee() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class Tee(io.TextIOBase):
        def __init__(self, target, buffer):
            self.target = target
            self.buffer = buffer

        def write(self, data):
            self.target.write(data)
            self.buffer.write(data)
            return len(data)

        def flush(self):
            self.target.flush()
            self.buffer.flush()

    sys.stdout = Tee(original_stdout, stream)
    sys.stderr = Tee(original_stderr, stream)
    try:
        yield stream
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def count_training_rows(train_data_path: Path) -> int:
    if not train_data_path.exists():
        return 0
    with train_data_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def append_history_csv(csv_path: Path, metrics_row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "model",
        "dataset_rows",
        "batch_size",
        "lr",
        "epochs",
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


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    resolved_path = Path(config_path or DEFAULT_CONFIG_PATH)
    if not resolved_path.exists():
        fallback_path = CONFIG_DIR / "bge_finetune_config.yaml"
        if fallback_path.exists() and fallback_path != resolved_path:
            resolved_path = fallback_path
        else:
            return {}
    with resolved_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded if isinstance(loaded, dict) else {}


def get_nested(config: Dict[str, Any], *keys: str, default: Optional[Any] = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


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


def infer_model_key(base_model: str) -> str:
    lowered = base_model.lower()
    if "bge-large" in lowered:
        return "bge-large"
    if "bge-base" in lowered:
        return "bge-base"
    return "bge"


# 注释掉的部分恢复后就是用hardneg版本的jsonl作训练集，也会生成相应待hardneg后缀的模型权重文件和日志结果文件
def build_experiment_name(base_output_dir: Path, batch_size: int, num_epochs: int, learning_rate: float) -> str:
# def build_experiment_name(base_output_dir: Path, batch_size: int, num_epochs: int, learning_rate: float, train_data: Optional[Path] = None) -> str:
    lr_str = f"{learning_rate:.0e}".replace("-0", "-").replace("+0", "")
    base_name = f"bs{batch_size}_ep{num_epochs}_lr{lr_str}"
    # train_name = str(train_data or "").lower()
    # if "hardneg" in train_name:
    #     base_name = f"{base_name}_hardneg"

    index = 1
    while True:
        # candidate_name = f"exp{index:03d}_{base_name}"
        candidate_name = f"{base_name}"
        if not (base_output_dir / candidate_name).exists():
            return candidate_name
        index += 1


def main() -> None:
    config = load_config()
    model_config = get_nested(config, "model", default={})
    dataset_config = get_nested(config, "dataset", default={})
    training_config = get_nested(config, "training", default={})
    embedding_config = get_nested(config, "embedding", default={})
    logging_config = get_nested(config, "logging", default={})

    base_model = model_config.get("name") or model_config.get("base_model") or "BAAI/bge-base-zh-v1.5"
    model_key = infer_model_key(base_model)
    train_data = Path(
        dataset_config.get("train_data")
        or "dataset/tool_calling_trace/train/intent_train_flagembedding.jsonl"
        # or "dataset/tool_calling_trace/train/intent_train_flagembedding_hardneg.jsonl"
    )
    batch_size = int(training_config.get("batch_size", 32))
    num_epochs = int(training_config.get("epochs", 3))
    learning_rate = float(training_config.get("learning_rate", 1e-5))
    seed = int(training_config.get("seed", 42))
    save_strategy = str(training_config.get("save_strategy", "epoch"))
    save_total_limit = int(training_config.get("save_total_limit", 3))
    warmup_ratio = training_config.get("warmup_ratio")
    logging_steps = int(logging_config.get("logging_steps", 10))
    save_loss_curve = bool(logging_config.get("save_loss_curve", True))
    query_instruction = embedding_config.get(
        "query_instruction",
        "Represent this sentence for searching relevant tool descriptions: ",
    )
    query_max_len = int(embedding_config.get("query_max_len", 128))
    passage_max_len = int(embedding_config.get("passage_max_len", 256))
    train_group_size = int(embedding_config.get("train_group_size", 1))
    normalize_embeddings = bool(embedding_config.get("normalize_embeddings", True))

    base_output_dir = Path(model_config.get("output_dir") or f"model/my-{model_key}")
    experiment_name = build_experiment_name(base_output_dir, batch_size, num_epochs, learning_rate)
    # experiment_name = build_experiment_name(base_output_dir, batch_size, num_epochs, learning_rate, train_data)
    model_output_dir = base_output_dir / experiment_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    log_base_dir = Path("outputs/rag/finetune") / model_key / experiment_name
    log_base_dir.mkdir(parents=True, exist_ok=True)

    loss_path = log_base_dir / "training_loss.csv"
    loss_plot_path = log_base_dir / "training_loss.png"
    experiment_history_path = Path("outputs/rag/finetune") / model_key / "experiment_history.csv"
    experiment_history_path.parent.mkdir(parents=True, exist_ok=True)

    for stale_path in [
        log_base_dir / "training_run.log",
        log_base_dir / "training_metrics.csv",
        log_base_dir / "training_metrics_summary.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    try:
        from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
            EncoderOnlyEmbedderM3DataArguments,
            EncoderOnlyEmbedderM3ModelArguments,
            EncoderOnlyEmbedderM3Runner,
            EncoderOnlyEmbedderM3TrainingArguments,
        )
        import FlagEmbedding.finetune.embedder.encoder_only.m3.runner as m3_runner
        from FlagEmbedding.finetune.embedder.encoder_only.m3.trainer import (
            EncoderOnlyEmbedderM3Trainer,
        )
    except ImportError as exc:
        raise SystemExit(
            "当前环境里的 FlagEmbedding 版本不支持此前的训练接口，请确认安装包版本是否正确。"
        ) from exc

    print(f"开始微调模型：{model_key}")
    print(f"基础模型：{base_model}")
    print(f"训练数据：{train_data}")
    print(f"模型保存目录：{model_output_dir}")
    print(f"日志与指标目录：{log_base_dir}")

    model_args = EncoderOnlyEmbedderM3ModelArguments(
        model_name_or_path=base_model,
        trust_remote_code=False,
        cache_dir=None,
    )
    data_args = EncoderOnlyEmbedderM3DataArguments(
        train_data=[str(train_data)],
        query_instruction_for_retrieval=query_instruction,
        query_max_len=query_max_len,
        passage_max_len=passage_max_len,
        train_group_size=train_group_size,
        shuffle_ratio=0.0,
    )
    training_args = EncoderOnlyEmbedderM3TrainingArguments(
        output_dir=str(model_output_dir),
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=1000,
        save_total_limit=save_total_limit,
        report_to="none",
        fp16=False,
        bf16=False,
        do_train=True,
        do_eval=False,
        normalize_embeddings=normalize_embeddings,
        sentence_pooling_method="cls",
        logging_dir=str(log_base_dir),
        seed=seed,
    )
    if warmup_ratio is not None:
        training_args.warmup_ratio = warmup_ratio
    training_args.overwrite_output_dir = True

    class CompatibleEncoderOnlyEmbedderM3Trainer(EncoderOnlyEmbedderM3Trainer):
        def __init__(self, *args, processing_class=None, **kwargs):
            self.tokenizer = processing_class
            super().__init__(*args, processing_class=processing_class, **kwargs)

    m3_runner.EncoderOnlyEmbedderM3Trainer = CompatibleEncoderOnlyEmbedderM3Trainer

    runner = EncoderOnlyEmbedderM3Runner(model_args, data_args, training_args)

    start_time = time.time()
    with capture_and_tee():
        train_output = runner.trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        runner.trainer.save_model()
    elapsed_seconds = int(time.time() - start_time)

    training_metrics = getattr(train_output, "metrics", {}) or {}
    if not isinstance(training_metrics, dict):
        training_metrics = dict(training_metrics)

    loss_series = []
    for entry in getattr(runner.trainer.state, "log_history", []):
        if isinstance(entry, dict) and isinstance(entry.get("loss"), (int, float)):
            loss_series.append(float(entry["loss"]))

    final_loss = training_metrics.get("train_loss")
    if final_loss is None and loss_series:
        final_loss = loss_series[-1]
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
        "model": model_key,
        "dataset_rows": count_training_rows(train_data),
        "batch_size": batch_size,
        "lr": learning_rate,
        "epochs": num_epochs,
        "global_step": int(getattr(train_output, "global_step", 0) or 0),
        "runtime_seconds": training_metrics.get("train_runtime", elapsed_seconds),
        "samples_per_second": training_metrics.get("train_samples_per_second"),
        "steps_per_second": training_metrics.get("train_steps_per_second"),
        "final_loss": "" if final_loss is None else f"{final_loss:.6f}",
        "final_learning_rate": "" if final_learning_rate is None else f"{final_learning_rate:.6e}",
        "seed": seed,
        "output_dir": str(model_output_dir),
    }
    append_history_csv(experiment_history_path, experiment_row)

    if save_loss_curve and loss_series:
        with loss_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for idx, value in enumerate(loss_series, start=1):
                writer.writerow([idx, value])

        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1, len(loss_series) + 1), loss_series, marker="o", linewidth=1.2)
        plt.title(f"{model_key} training loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=200)
        plt.close()

    print(f"微调完成，模型已保存到：{model_output_dir}")
    print(f"实验汇总已保存到：{experiment_history_path}")
    print(f"训练 loss 曲线已保存到：{loss_path}")
    print(f"训练 loss 图片已保存到：{loss_plot_path}")


if __name__ == "__main__":
    main()
