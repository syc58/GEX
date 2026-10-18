"""Object grounding metrics for grounded explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


SEMANTIC_THRESHOLD = 0.75
IOU_THRESHOLD = 0.5

_BOX_CONTENT = (
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_REF_BOX_RE = re.compile(
    rf"<ref>(?P<label>.*?)</ref>\s*<box>\s*(?P<box>{_BOX_CONTENT})\s*</box>",
    flags=re.IGNORECASE | re.DOTALL,
)
_BOX_RE = re.compile(_BOX_CONTENT)


@dataclass(frozen=True)
class Grounding:
    label: str
    box: tuple[float, float, float, float]


class SemanticMatcher:
    """Lazy semantic label matcher backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = SEMANTIC_THRESHOLD,
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.enabled = enabled
        self._model = None
        self._cosine_similarity = None

    def match(self, prediction: str, reference: str) -> bool:
        prediction = normalize_label(prediction)
        reference = normalize_label(reference)
        if not prediction or not reference:
            return False
        if prediction == reference or prediction in reference or reference in prediction:
            return True
        if not self.enabled:
            return False

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity

            self._model = SentenceTransformer(self.model_name)
            self._cosine_similarity = cosine_similarity

        embeddings = self._model.encode([prediction, reference])
        score = self._cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(score) >= self.threshold


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().lower())


def parse_box(text: str) -> tuple[float, float, float, float]:
    match = _BOX_RE.search(text)
    if not match:
        raise ValueError(f"Invalid box format: {text}")
    return tuple(float(match.group(i)) for i in range(1, 5))


def parse_groundings(
    explanation: str,
    key_objects: Iterable[str] | None = None,
) -> list[Grounding]:
    """Extract grounded objects from either tagged or released plain format."""

    explanation = explanation or ""
    tagged = [
        Grounding(normalize_label(match.group("label")), parse_box(match.group("box")))
        for match in _REF_BOX_RE.finditer(explanation)
    ]
    if tagged:
        return tagged

    if key_objects:
        groundings: list[Grounding] = []
        for label in key_objects:
            label_text = str(label).strip()
            if not label_text:
                continue
            pattern = re.compile(
                rf"{re.escape(label_text)}\s*(?P<box>{_BOX_CONTENT})",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(explanation):
                groundings.append(
                    Grounding(normalize_label(label_text), parse_box(match.group("box")))
                )
        if groundings:
            return groundings

    return parse_plain_groundings(explanation)


def parse_plain_groundings(explanation: str) -> list[Grounding]:
    """Best-effort fallback for text like 'red car [1, 2, 3, 4]'."""

    groundings: list[Grounding] = []
    for match in _BOX_RE.finditer(explanation or ""):
        prefix = explanation[: match.start()].strip()
        prefix = re.sub(r"[\s,.;:()]+$", "", prefix)
        label_match = re.search(r"([A-Za-z][A-Za-z0-9'\- ]{0,80})$", prefix)
        if not label_match:
            continue
        label = normalize_label(label_match.group(1))
        groundings.append(Grounding(label, parse_box(match.group(0))))
    return groundings


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0

    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def grounding_f1_for_sample(
    predictions: list[Grounding],
    references: list[Grounding],
    matcher: SemanticMatcher,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[float, float, float]:
    if not predictions and not references:
        return 1.0, 1.0, 1.0
    if not predictions or not references:
        return 0.0, 0.0, 0.0

    candidate_pairs = sorted(
        (
            (pred_idx, ref_idx, box_iou(pred.box, ref.box))
            for pred_idx, pred in enumerate(predictions)
            for ref_idx, ref in enumerate(references)
        ),
        key=lambda item: item[2],
        reverse=True,
    )

    matched_predictions: set[int] = set()
    matched_references: set[int] = set()
    for pred_idx, ref_idx, overlap in candidate_pairs:
        if pred_idx in matched_predictions or ref_idx in matched_references:
            continue
        if overlap < iou_threshold:
            continue
        if not matcher.match(predictions[pred_idx].label, references[ref_idx].label):
            continue
        matched_predictions.add(pred_idx)
        matched_references.add(ref_idx)

    true_positive = len(matched_predictions)
    precision = true_positive / len(predictions)
    recall = true_positive / len(references)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_grounding_metrics(
    predictions: list[dict],
    references_by_qid: dict[str, dict],
    *,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    use_semantic_matching: bool = True,
) -> dict:
    matcher = SemanticMatcher(
        threshold=semantic_threshold,
        enabled=use_semantic_matching,
    )

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    skipped = 0

    for prediction in predictions:
        qid = str(prediction.get("qid", ""))
        reference = references_by_qid.get(qid)
        if not qid or reference is None:
            skipped += 1
            continue

        predicted_groundings = parse_groundings(
            prediction.get("explanation", ""),
            prediction.get("key_objects"),
        )
        reference_groundings = parse_groundings(
            reference.get("explanation", ""),
            reference.get("key_objects"),
        )
        precision, recall, f1 = grounding_f1_for_sample(
            predicted_groundings,
            reference_groundings,
            matcher,
            iou_threshold=iou_threshold,
        )
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    matched = len(f1_scores)
    if matched == 0:
        return {"matched": 0, "skipped": skipped}

    return {
        "matched": matched,
        "skipped": skipped,
        "grounding_precision": sum(precision_scores) / matched,
        "grounding_recall": sum(recall_scores) / matched,
        "grounding_f1": sum(f1_scores) / matched,
    }
