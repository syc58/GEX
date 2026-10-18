import json
import re
import os
import base64
from collections import defaultdict
from openai import OpenAI

splits = ["train"]
base_dir = "../dataset"
IMAGE_ROOT = "../DOCCI/images"
client = OpenAI(api_key="xxxxx")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def judge_with_image(image_path, question, answer, explanation):
    prompt = f"""
Given an image, a question, an answer, and an explanation, decide whether the explanation is blatantly inconsistent with the image or clearly fails to support the answer.

Definitions:
- Supported: The explanation is broadly consistent with the image and could reasonably justify the answer. Do not require fine-grained verification.
- Not Supported: The explanation relies on key visual details that are absent in the image, or fails to connect to the answer.
- Contradicted: The explanation clearly conflicts with visible evidence.

Guidelines:
- Be tolerant to blur, occlusion, distance, and ambiguous regions.
- Only penalize details that are clearly wrong or clearly not present.
- Low-risk commonsense is allowed if it matches visible cues.
- If unsure and there is no clear conflict, prefer Supported over Not Supported.

Output:
Supported / Not Supported / Contradicted

Question: {question}
Answer: {answer}
Explanation: {explanation}
""".strip()
    base64_image = encode_image(image_path)
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )
    return response.output_text.strip()


for split in splits:
    input_file = os.path.join(base_dir, split, "03_EAC_Check.json")
    output_file = os.path.join(base_dir, split, "04_MC_Check.json")
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
        image_path = os.path.join(IMAGE_ROOT, image)
        if not os.path.exists(image_path):
            continue
        total = len(qas)
        kept = 0
        removed = 0
        for qa in qas:
            label = judge_with_image(
                image_path=image_path,
                question=qa["question"],
                answer=qa["answer"],
                explanation=qa["explanation"],
            )
            if label == "Supported":
                filtered_results.append(qa)
                kept += 1
            else:
                removed += 1
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=4, ensure_ascii=False)
