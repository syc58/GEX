"""Run GEX evaluation metrics by matching predictions to references with qid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics.grounding_metrics import compute_grounding_metrics
from metrics.lave_metrics import compute_lave_metrics
from metrics.text_metrics import compute_text_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_PATH = PROJECT_ROOT / "dataset" / "test.json"
PREDICTION_PATHS: list[str | Path] = []

METRICS = ("text", "grounding")
IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
LAVE_CACHE_PATH = Path(__file__).resolve().parent / "lave_scores_cache.json"
LAVE_LIMIT = None
JUDGE_MODEL = "gpt-5"

INCLUDE_BERT_SCORE = True
USE_SEMANTIC_GROUNDING = True
OPENAI_API_KEY = None
OPENAI_BASE_URL = None
OUTPUT_PATH = None


def load_json(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def index_by_qid(items: list[dict], source_name: str) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for row_number, item in enumerate(items, start=1):
        qid = str(item.get("qid", "")).strip()
        if not qid:
            raise ValueError(f"Missing qid in {source_name} at row {row_number}")
        if qid in index:
            raise ValueError(f"Duplicate qid in {source_name}: {qid}")
        index[qid] = item
    return index


def prediction_name(path: str | Path) -> str:
    path = Path(path)
    if path.name == "process_predictions.json":
        return path.parent.name
    return path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions",
        nargs="*",
        help="Prediction JSON files. If omitted, PREDICTION_PATHS is used.",
    )
    return parser.parse_args()


def selected_metrics(raw_metrics: tuple[str, ...]) -> set[str]:
    if "all" in raw_metrics:
        return {"text", "grounding", "lave"}
    return set(raw_metrics)


def main() -> None:
    args = parse_args()
    metrics = selected_metrics(METRICS)
    prediction_paths = [Path(path) for path in (args.predictions or PREDICTION_PATHS)]

    if not prediction_paths:
        raise ValueError("Provide prediction files as arguments or set PREDICTION_PATHS.")
    if "lave" in metrics and not Path(IMAGE_DIR).exists():
        raise ValueError(f"IMAGE_DIR does not exist: {IMAGE_DIR}")

    references = load_json(GT_PATH)
    references_by_qid = index_by_qid(references, str(GT_PATH))

    prediction_sets: dict[str, list[dict]] = {}
    prediction_indexes: dict[str, dict[str, dict]] = {}
    for path in prediction_paths:
        name = prediction_name(path)
        predictions = load_json(path)
        prediction_sets[name] = predictions
        prediction_indexes[name] = index_by_qid(predictions, str(path))

    results: dict[str, dict] = {}
    for name, predictions in prediction_sets.items():
        model_results: dict[str, dict] = {}
        if "text" in metrics:
            model_results["text"] = compute_text_metrics(
                predictions,
                references_by_qid,
                include_bert_score=INCLUDE_BERT_SCORE,
            )
        if "grounding" in metrics:
            model_results["grounding"] = compute_grounding_metrics(
                predictions,
                references_by_qid,
                use_semantic_matching=USE_SEMANTIC_GROUNDING,
            )
        results[name] = model_results

    if "lave" in metrics:
        lave_results = compute_lave_metrics(
            prediction_indexes,
            references_by_qid,
            image_dir=IMAGE_DIR,
            cache_path=LAVE_CACHE_PATH,
            judge_model=JUDGE_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            limit=LAVE_LIMIT,
        )
        for name, metrics_for_model in lave_results.items():
            results.setdefault(name, {})["lave"] = metrics_for_model

    output_text = json.dumps(results, ensure_ascii=False, indent=2)
    print(output_text)

    if OUTPUT_PATH:
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
