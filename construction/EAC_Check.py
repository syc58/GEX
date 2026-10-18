import json
import re
import os
from openai import OpenAI
from collections import defaultdict

splits = ["train"]
base_dir = "../dataset"
client = OpenAI(api_key="xxxxx")


def judge_consistency(question, answer, explanation):
    prompt = f"""
Given a question, an answer, and an explanation, determine whether the explanation is consistent with and supports the given answer.

Definitions:
- Supported: The explanation logically supports the answer and is consistent with the question. It provides a reasonable justification for why the answer is correct, possibly using common-sense or causal reasoning.
- Not Supported: The explanation does not clearly justify the answer, is irrelevant, incomplete, or supports a different answer.
- Contradicted: The explanation conflicts with the given answer.

Guidelines:
1) Judge support with respect to the question: the explanation must address the question, not just discuss the topic.
2) Extract key claims in the explanation and check they lead to the given answer (not another plausible answer).
3) If the explanation could fit multiple answers or lacks a clear link, choose Not Supported.
4) Use commonsense/causal reasoning only if it directly connects the explanation to the given answer.
5) Prefer Contradicted only when the explanation clearly points to a different answer; otherwise use Not Supported.

Output format:
Supported / Not Supported / Contradicted

Question: {question}
Answer: {answer}
Explanation: {explanation}
""".strip()
    response = client.responses.create(
        model="gpt-5", reasoning={"effort": "medium"}, input=prompt
    )
    return response.output_text.strip()


for split in splits:
    input_file = os.path.join(base_dir, split, "02_DEC_Check.json")
    output_file = os.path.join(base_dir, split, "03_EAC_Check.json")
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
        total = len(qas)
        kept = 0
        removed = 0
        for qa in qas:
            label = judge_consistency(qa["question"], qa["answer"], qa["explanation"])
            if label == "Supported":
                filtered_results.append(qa)
                kept += 1
            else:
                removed += 1
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=4, ensure_ascii=False)
