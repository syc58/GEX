# GEX: A Grounded EXplanation Benchmark for Explanatory Visual Question Answering

This repository provides the released dataset and evaluation code for **GEX**, a
benchmark for explanatory visual question answering with grounded explanations.
GEX evaluates whether a model can answer a visual question, explain the answer,
and ground key visual evidence in the image.

## Dataset

The released dataset is stored in `dataset/`:

- `dataset/train.json`: 18,000 training samples.
- `dataset/test.json`: 6,000 test samples.

Images are not included in this repository, download the original DOCCI images from the official
DOCCI project page: https://google.github.io/docci/ (`DOCCI Images`, 7.1 GB).
The dataset is also available through Hugging Face Datasets at
https://huggingface.co/datasets/google/docci.



## Construction

The anonymized data construction pipeline is stored in `construction/`. It
contains the scripts used to generate candidate samples, run consistency checks,
localize visual evidence, verify evidence boxes, and produce the final grounded
output. API keys in these scripts are replaced with `xxxxx`.

Run the scripts in order:

```text
construction/sample_generation.py
construction/DEC_Check.py
construction/EAC_Check.py
construction/MC_Check.py
construction/evidence_localization.py
construction/evidence_verfication.py
construction/output.py
```

The stage outputs are named sequentially from `01_sample_generation.json` to
`07_output.json`.

## Training and Prediction

The LLaMA-Factory training and prediction configurations are stored in
`training/`. The Qwen3.5-4B LoRA SFT setup is provided as YAML files with thin
shell wrappers:

```bash
bash training/run_sft.sh
bash training/run_predict.sh
```

The training split uses `gex_train`, and prediction uses `gex_test`.

## Evaluation

The evaluation code is stored in `evaluation/`.

- `evaluation/evaluate.py`: main script for running all selected metrics. It
  matches predictions to references by `qid`.
- `evaluation/metrics/text_metrics.py`: explanation quality metrics, including
  BLEU-4, ROUGE-L, METEOR, and BERTScore-F1.
- `evaluation/metrics/grounding_metrics.py`: grounding metric Object-aware Grounding F1.
- `evaluation/metrics/lave_metrics.py`: LAVE-style answer evaluation with a
  vision-language judge model.

By default, `evaluate.py` runs the text and grounding metrics on
`dataset/test.json`. LAVE is implemented but not enabled by default because it
requires image files and access to a judge model API.

## Prediction Format

Prediction files should be JSON lists. Each prediction must contain `qid`,
`answer`, and `explanation`. The `qid` is used to match each prediction with the
corresponding reference sample.

Example:

```json
[
  {
    "qid": "538179",
    "answer": "Ancient seabed",
    "explanation": "The fossil shapes [620, 560, 3480, 2800] on the rock surface [10, 8, 4010, 3000] suggest an ancient seabed.",
    "key_objects": ["fossil shapes", "rock surface"]
  }
]
```

The grounding metric supports both the released plain format above and the
tagged format:

```text
<ref>fossil shapes</ref><box>[620, 560, 3480, 2800]</box>
```

## Running Evaluation

Install the metric dependencies used by the selected metrics, then run:

```powershell
python evaluation/evaluate.py path/to/predictions.json
```

The script prints a JSON object with one result block per prediction file. For
multiple prediction files:

```powershell
python evaluation/evaluate.py path/to/model_a.json path/to/model_b.json
```

To change the selected metrics or default paths, edit the configuration values
at the top of `evaluation/evaluate.py`, such as:

```python
METRICS = ("text", "grounding")
GT_PATH = PROJECT_ROOT / "dataset" / "test.json"
```

To enable LAVE, include `"lave"` in `METRICS`. The default `IMAGE_DIR` is
`dataset/images`. The OpenAI-compatible API key can be supplied through the environment
or by setting `OPENAI_API_KEY` in `evaluate.py`.
