import json
import re
import os
from collections import defaultdict
from openai import OpenAI

splits = ["train"]
base_dir = "../dataset"
client = OpenAI(api_key="xxxxx")
descriptions_file = "../DOCCI/docci_descriptions.json"
with open(descriptions_file, "r", encoding="utf-8") as f:
    descriptions = json.load(f)
image2desc = {item["image_file"]: item["description"] for item in descriptions}
BASE_PROMPT = """
Given a description and multiple explanations, determine whether each explanation is reasonably supported by the description.

Definitions:
- Supported: The explanation is consistent with the description and can be reasonably inferred from it, including through common-sense knowledge or causal reasoning that does not contradict the description.
- Not Mentioned: The explanation introduces information or conclusions that are not reasonably inferable from the description, even with common-sense or causal reasoning.
- Contradicted: The explanation conflicts with information explicitly stated in the description.

Guidelines:
- Each explanation is judged independently; do not compare explanations.
- Explanations may include causal language and world knowledge.
- Common-sense inferences are allowed as long as they are natural extensions of the visual scene.
- Do not require that every detail of the explanation be explicitly stated in the description.
- Reject explanations that rely on speculative, highly specific, or unsupported assumptions.

Output format:
Explanation 1: Supported / Not Mentioned / Contradicted
Explanation 2: Supported / Not Mentioned / Contradicted
...
""".strip()
for split in splits:
    input_file = os.path.join(base_dir, split, "01_sample_generation.json")
    output_file = os.path.join(base_dir, split, "02_DEC_Check.json")
    if not os.path.exists(input_file):
        continue
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                for qa in existing_results:
                    processed_images.add(qa["image"])
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []
    with open(input_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    image2qas = defaultdict(list)
    for qa in qa_data:
        if qa["image"] not in processed_images:
            image2qas[qa["image"]].append(qa)
    if len(image2qas) == 0:
        continue
    filtered_results = existing_results.copy()
    for (idx, (image, qas)) in enumerate(image2qas.items(), start=1):
        description = (
            image2desc[image]
            if "image2desc" in locals() and image in image2desc
            else ""
        )
        prompt = f"""
{BASE_PROMPT}

Description:
{description}
""".strip()
        for (i, qa) in enumerate(qas, start=1):
            prompt += f"""

Explanation {i}: {qa['explanation']}"""
        response = client.responses.create(
            model="gpt-5", reasoning={"effort": "medium"}, input=prompt
        )
        output_text = response.output_text
        labels = []
        for line in output_text.splitlines():
            match = re.search(
                "Explanation\\s+(\\d+):\\s*(Supported|Not Mentioned|Contradicted)", line
            )
            if match:
                labels.append(match.group(2))
        if len(labels) != len(qas):
            continue
        total = len(qas)
        removed = 0
        kept = 0
        for (qa, label) in zip(qas, labels):
            if label == "Supported":
                filtered_results.append(qa)
                kept += 1
            else:
                removed += 1
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=4, ensure_ascii=False)
