"""LAVE-style answer evaluation with a vision-language judge."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


DEFAULT_JUDGE_MODEL = "gpt-5"

TASK_DESCRIPTION = """You are given an image, a question, a gold-standard reference answer written by experts, and candidate answers from one or more models.

Evaluate how well each candidate answer answers the question.

Use the image and reference answer as guidance, but do not require exact wording. Judge semantic correctness.

Use this 1-3 scale:
1 = incorrect, irrelevant, or contradicts the image/question/reference answer
2 = partially correct, too vague, incomplete, or ambiguous
3 = correct; semantically answers the question even if the wording differs

For yes/no questions, judge the meaning rather than exact words. For open-ended questions, accept synonyms, paraphrases, or more specific answers when they are consistent with the image and reference answer. If a candidate answer is much longer than the reference answer but semantically correct, cap the score at 2.

Give a brief rationale, then output exactly one JSON object mapping each candidate index to its integer rating. The final JSON format must be:
{"candidate_1": score_1, "candidate_2": score_2}
"""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def normalize_references(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [normalize_text(item) for item in answer if normalize_text(item)]
    text = normalize_text(answer)
    return [text] if text else []


def load_cache(cache_path: str | Path | None) -> dict[str, dict[str, int]]:
    if cache_path is None:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_cache(cache: dict[str, dict[str, int]], cache_path: str | Path | None) -> None:
    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
        file.write("\n")


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    with path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_lave_prompt(
    question: str,
    references: list[str],
    candidates: dict[str, str],
) -> str:
    references_text = "; ".join(references)
    candidates_text = "\n".join(f"{key}: {value}" for key, value in candidates.items())
    return (
        f"{TASK_DESCRIPTION}\n\n"
        f"Question: {question}\n"
        f"Reference answer: {references_text}\n"
        f"Candidate answers:\n{candidates_text}\n"
        f"Output:"
    )


def extract_json_object(text: str) -> dict:
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in judge response:\n{text}")
    return json.loads(matches[-1])


def parse_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 1
    return score if score in {1, 2, 3} else 1


def make_openai_client(api_key: str | None = None, base_url: str | None = None):
    from openai import OpenAI

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def score_answers_with_lave(
    *,
    question: str,
    references: list[str],
    model_answers: dict[str, str],
    image_path: str | Path,
    client,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, int]:
    model_names = list(model_answers.keys())
    anonymized_candidates = {
        f"candidate_{index + 1}": normalize_text(model_answers[name])
        for index, name in enumerate(model_names)
    }
    prompt = build_lave_prompt(question, references, anonymized_candidates)
    response = client.responses.create(
        model=judge_model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }
        ],
    )

    parsed = extract_json_object(response.output_text.strip())
    return {
        name: parse_score(parsed.get(f"candidate_{index + 1}", 1))
        for index, name in enumerate(model_names)
    }


def compute_lave_metrics(
    model_predictions_by_qid: dict[str, dict[str, dict]],
    references_by_qid: dict[str, dict],
    *,
    image_dir: str | Path,
    cache_path: str | Path | None = "lave_scores_cache.json",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    limit: int | None = None,
) -> dict[str, dict]:
    if not model_predictions_by_qid:
        raise ValueError("No prediction files were provided for LAVE evaluation.")

    model_names = list(model_predictions_by_qid.keys())
    shared_qids = set(references_by_qid)
    for predictions_by_qid in model_predictions_by_qid.values():
        shared_qids &= set(predictions_by_qid)

    qids = sorted(shared_qids, key=lambda item: int(item) if item.isdigit() else item)
    if limit is not None:
        qids = qids[:limit]

    cache = load_cache(cache_path)
    pending_qids = [
        qid
        for qid in qids
        if qid not in cache or any(name not in cache[qid] for name in model_names)
    ]

    client = make_openai_client(api_key=api_key, base_url=base_url)
    image_root = Path(image_dir)
    skipped = 0

    for index, qid in enumerate(pending_qids, start=1):
        reference = references_by_qid[qid]
        question = normalize_text(reference.get("question"))
        references = normalize_references(reference.get("answer"))
        image_name = normalize_text(reference.get("image"))
        image_path = image_root / image_name

        if not question or not references or not image_name or not image_path.exists():
            skipped += 1
            continue

        model_answers = {
            name: normalize_text(model_predictions_by_qid[name][qid].get("answer"))
            for name in model_names
        }
        scores = score_answers_with_lave(
            question=question,
            references=references,
            model_answers=model_answers,
            image_path=image_path,
            client=client,
            judge_model=judge_model,
        )

        cache.setdefault(qid, {}).update(scores)
        save_cache(cache, cache_path)

        if index % 100 == 0 or index == len(pending_qids):
            print(f"LAVE progress: {index}/{len(pending_qids)}")

    results: dict[str, dict] = {}
    for name in model_names:
        scores = [
            cache[qid][name]
            for qid in qids
            if qid in cache and name in cache[qid]
        ]
        if not scores:
            results[name] = {"matched": 0, "skipped": skipped}
            continue

        distribution = {1: 0, 2: 0, 3: 0}
        for score in scores:
            distribution[score] += 1

        average = sum(scores) / len(scores)
        results[name] = {
            "matched": len(scores),
            "skipped": skipped,
            "lave": average,
            "lave_0_1": (average - 1) / 2,
            "score_1": distribution[1],
            "score_2": distribution[2],
            "score_3": distribution[3],
        }

    return results
