"""Text-generation metrics for explanations."""

from __future__ import annotations

import re


_BOX_RE = re.compile(
    r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,"
    r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]"
)


def normalize_explanation_text(text: str) -> str:
    """Remove grounding markup and coordinates before text-only evaluation."""

    text = text or ""
    text = re.sub(
        r"<ref>(.*?)</ref>\s*<box>\s*\[[^\]]+\]\s*</box>",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"</?ref>|</?box>", "", text, flags=re.IGNORECASE)
    text = _BOX_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def ensure_nltk_data() -> None:
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def compute_text_metrics(
    predictions: list[dict],
    references_by_qid: dict[str, dict],
    *,
    include_bert_score: bool = True,
    download_nltk: bool = True,
) -> dict:
    if download_nltk:
        ensure_nltk_data()

    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    references_for_bleu: list[list[list[str]]] = []
    hypotheses_for_bleu: list[list[str]] = []
    rouge_scores: list[float] = []
    meteor_scores: list[float] = []
    prediction_texts: list[str] = []
    reference_texts: list[str] = []
    skipped = 0

    for prediction in predictions:
        qid = str(prediction.get("qid", ""))
        reference = references_by_qid.get(qid)
        if not qid or reference is None:
            skipped += 1
            continue

        prediction_text = normalize_explanation_text(prediction.get("explanation", ""))
        reference_text = normalize_explanation_text(reference.get("explanation", ""))
        if not prediction_text or not reference_text:
            skipped += 1
            continue

        prediction_tokens = prediction_text.split()
        reference_tokens = reference_text.split()

        rouge_scores.append(
            rouge.score(reference_text, prediction_text)["rougeL"].fmeasure
        )
        meteor_scores.append(meteor_score([reference_tokens], prediction_tokens))
        references_for_bleu.append([reference_tokens])
        hypotheses_for_bleu.append(prediction_tokens)
        prediction_texts.append(prediction_text)
        reference_texts.append(reference_text)

    matched = len(prediction_texts)
    if matched == 0:
        return {"matched": 0, "skipped": skipped}

    results = {
        "matched": matched,
        "skipped": skipped,
        "bleu_4": corpus_bleu(
            references_for_bleu,
            hypotheses_for_bleu,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother,
        ),
        "rouge_l": sum(rouge_scores) / matched,
        "meteor": sum(meteor_scores) / matched,
    }

    if include_bert_score:
        from bert_score import score as bert_score

        _, _, bert_f1 = bert_score(
            prediction_texts,
            reference_texts,
            lang="en",
            verbose=False,
        )
        results["bertscore_f1"] = sum(bert_f1.tolist()) / matched

    return results
