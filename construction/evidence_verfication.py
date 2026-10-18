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


def judge_with_image(image_path, visual_objects):
    prompt = f"""
Given an image and a list of visual objects, each with one bounding box, perform a conservative sanity-check for grounding quality.

Task:
- Decide whether the provided boxes reasonably match the named objects in the image.
- Ignore minor localization errors, slightly loose/tight boxes, blur, partial occlusion, and small scale mismatch.
- Only flag a sample as bad when the mismatch is clear.
- Do not require perfect box quality.

Definitions:
- Match: All objects have a plausible box, or only one object is slightly doubtful in a crowded or ambiguous image, and there is no obvious object-name confusion.
- Partial: Some objects match, but at least one object fails or is very doubtful.
- Mismatch: Many objects fail, or boxes clearly correspond to different regions/objects than their names, or multiple different objects collapse to the same wrong region.

Guidelines:
- Judge each object independently first, then make an overall decision.
- A box is acceptable if it overlaps the intended object reasonably, even if not perfectly tight.
- If the named object is not actually visible, or the box is supported only by very weak evidence, do not count it as a valid match.
- For filtering, prefer Partial over Match when uncertain.
- Consider Mismatch only when there is obvious conflict (e.g., the box is on a completely unrelated region).
- If different object names share nearly identical boxes and they are clearly different entities, treat as Mismatch.

Output format (exactly one line):
Match / Partial / Mismatch

Visual objects: {visual_objects}
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


def has_low_score_sample(qa, threshold=0.396):
    if "bounding_boxes" not in qa:
        return False
    for (obj_name, box_data) in qa["bounding_boxes"].items():
        scores = box_data.get("scores", [])
        for score in scores:
            if score < threshold:
                return True
    return False


def build_visual_objects(qa):
    visual_objects = []
    if "bounding_boxes" in qa:
        for (obj_name, box_data) in qa["bounding_boxes"].items():
            boxes = box_data.get("boxes", [])
            scores = box_data.get("scores", [])
            for (i, box) in enumerate(boxes):
                score = scores[i] if i < len(scores) else 0.0
                visual_objects.append({"object": obj_name, "box": box})
    return visual_objects


for split in splits:
    input_file = os.path.join(base_dir, split, "05_evidence_localization.json")
    output_file = os.path.join(base_dir, split, "06_evidence_verification.json")
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
    unprocessed_qas = []
    for qa in qa_data:
        if qa["image"] not in processed_images:
            unprocessed_qas.append(qa)
    if len(unprocessed_qas) == 0:
        continue
    filtered_results = existing_results.copy()
    total_samples = len(unprocessed_qas)
    kept_count = 0
    removed_by_score = 0
    removed_by_gpt = 0
    gpt_checked_count = 0
    for (idx, qa) in enumerate(unprocessed_qas, start=1):
        image = qa["image"]
        image_path = os.path.join(IMAGE_ROOT, image)
        if not os.path.exists(image_path):
            continue
        if has_low_score_sample(qa):
            gpt_checked_count += 1
            if not os.path.exists(image_path):
                removed_by_score += 1
                continue
            visual_objects = build_visual_objects(qa)
            label = judge_with_image(
                image_path=image_path, visual_objects=visual_objects
            )
            if label == "Match":
                filtered_results.append(qa)
                kept_count += 1
            else:
                removed_by_gpt += 1
        else:
            filtered_results.append(qa)
            kept_count += 1
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=4, ensure_ascii=False)
