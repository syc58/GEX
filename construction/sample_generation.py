import os
import re
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional
import random
from openai import OpenAI

DESCRIPTIONS_FILE = "../DOCCI/docci_descriptions.json"
OUTPUT_ROOT = "../dataset"
SPLIT_ORDER = ["train"]
API_KEY = "xxxxx"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3.5-plus"
SYSTEM_PROMPT = "You are constructing question-answer-explanation triples for an explanatory visual question answering dataset."
USER_PROMPT_TEMPLATE = """
Input:
One detailed natural image description containing multiple objects, attributes, spatial relations, materials, text, and environmental cues.

Task:
Create 4-6 diverse, challenging questions from the given description.
Each question must require non-trivial reasoning, such as:
- Comparing attributes, states, or positions across multiple objects
- Inferring cause, effect, function, or temporal sequence
- Integrating visual details with real-world knowledge
- Reasoning about absence, negation, or counterfactual scenarios
- Interpreting contextual, social, or cultural implications

Forbidden:
- Inventing objects, attributes, text, actions, or conditions not present in the scene.

Types:
Use the following reasoning types:
Attribute | Counting | Spatial Relationship | Scene | Text | World Knowledge
Prefer different types across questions; repeat only if unavoidable.

Forbidden meta/reporting language (applies to BOTH questions and explanations):
Do NOT use any wording that references a description or reading text, including:
described, as described, according to, in the description, the passage mentions, the text says/shows, depicted, mentioned, shown, seen, and so on.

Question rules:
- No more than 20 words.
- Ask as if directly observing a real image.
- Anchor questions to visible objects, layout, materials, text, or environmental conditions.
- No meta/reporting language.

Answer rules:
- A single word or phrase.
- Must be supported by visible evidence or clearly implied world knowledge.

Explanation rules:
- 15-40 tokens.
- No meta/reporting language.
- Final self-check: if any forbidden word/phrase appears, rewrite until none remain.

Key visual objects:
For each sample, output 2-4 key visual objects that support the explanation.
Rules:
- Each must be a concrete, visible entity/region that could be localized.
- Use short phrases (1-4 words), prefer head nouns with minimal modifiers (e.g., "abacus", "wood blocks", "cracked pavement").
- Do not include abstract concepts (e.g., neglect, maintenance, contrast, implication, atmosphere).
- Avoid semantic redundancy: do not include synonyms, near-duplicates, or hypernym pairs (e.g., "car" and "Volkswagen Beetle").
- Each key object must be explicitly mentioned in the explanation. Do not introduce new objects that are not mentioned in the explanation.

Output Format:
Return only a JSON array. Each element must contain:
{
  "question": "<string, no more than 20 words>",
  "answer": "<string, a single word or phrase>",
  "explanation": "<string, 15-40 tokens>",
  "type": "<one of: Attribute | Counting | Spatial Relationship | Scene | Text | World Knowledge>",
  "key_objects": ["<object1>", "<object2>", "..."]
}

Examples:
[
  {
    "question": "What traditional calculation tool contrasts with the modern technology word displayed?",
    "answer": "Abacus",
    "explanation": "The abacus represents manual calculation, contrasting with the blocks spelling COMPUTER representing modern digital technology concepts.",
    "type": "World Knowledge",
    "key_objects": ["abacus", "blocks"]
  },
  {
    "question": "What visual evidence most strongly supports interpreting the street as poorly maintained?",
    "answer": "Cracked pavement",
    "explanation": "Cracked pavement with grass, uneven faded paint, dirt, and graffiti together suggest long-term wear and lack of maintenance.",
    "type": "Scene",
    "key_objects": ["cracked pavement", "grass", "graffiti", "faded paint"]
  }
]
""".strip()


def load_descriptions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_split(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        split = item.get("split")
        if split in SPLIT_ORDER:
            grouped[split].append(item)
    return grouped


def extract_json_array(raw_text: str) -> List[Dict[str, Any]]:
    text = raw_text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    fenced = re.search("```(?:json)?\\s*(.*?)\\s*```", text, flags=re.S | re.I)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array start '[' found in model output.")
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("No balanced JSON array found in model output.")
    candidate = text[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, list):
        raise ValueError("Extracted JSON is not a list.")
    return obj


def call_model(client: OpenAI, description: str) -> (str, Optional[Any]):
    final_prompt = USER_PROMPT_TEMPLATE + "\nDescription:\n" + description
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
        ],
        extra_body={"enable_thinking": True},
        stream=True,
        stream_options={"include_usage": True},
    )
    answer_content = ""
    usage_info = None
    for chunk in completion:
        if not chunk.choices:
            usage_info = getattr(chunk, "usage", None)
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            answer_content += delta.content
    return (answer_content, usage_info)


def normalize_qa_list(
    generated_data: List[Dict[str, Any]], image_file: str, split: str
) -> List[Dict[str, Any]]:
    output_data = []
    for qa in generated_data:
        output_data.append(
            {
                "image": image_file,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "explanation": qa.get("explanation", ""),
                "type": qa.get("type", ""),
                "key_objects": qa.get("key_objects", []),
            }
        )
    return output_data


def process_split(
    client: OpenAI, split: str, items: List[Dict[str, Any]], output_root: str
) -> None:
    split_dir = os.path.join(output_root, split)
    os.makedirs(split_dir, exist_ok=True)
    output_file = os.path.join(split_dir, "01_sample_generation.json")
    split_results: List[Dict[str, Any]] = []
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            split_results = json.load(f)
        processed_images = {item.get("image") for item in split_results}
    remaining_items = [
        item for item in items if item.get("image_file") not in processed_images
    ]
    total_count = len(items)
    processed_count = len(remaining_items)
    skipped_count = total_count - processed_count
    for (idx, item) in enumerate(remaining_items, start=1):
        image_file = item.get("image_file", "")
        description = item.get("description", "")
        if not description:
            continue
        try:
            (answer_content, usage_info) = call_model(client, description)
            if usage_info:
                pass
            generated_data = extract_json_array(answer_content)
            one_image_qas = normalize_qa_list(generated_data, image_file, split)
            split_results.extend(one_image_qas)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(split_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            continue


def main() -> None:
    if API_KEY == "xxxxx":
        raise ValueError("Please set DASHSCOPE_API_KEY or replace API_KEY in script.")
    descriptions = load_descriptions(DESCRIPTIONS_FILE)
    grouped = group_by_split(descriptions)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for split in SPLIT_ORDER:
        items = grouped.get(split, [])
        split_dir = os.path.join(OUTPUT_ROOT, split)
        existing_generation_file = os.path.join(split_dir, "01_sample_generation.json")
        existing_images = set()
        if os.path.exists(existing_generation_file):
            with open(existing_generation_file, "r", encoding="utf-8") as f:
                existing_generation_data = json.load(f)
            existing_images = {item.get("image") for item in existing_generation_data}
        remaining_items = [
            item for item in items if item.get("image_file") not in existing_images
        ]
        sample_size = min(50, len(remaining_items))
        items = random.sample(remaining_items, sample_size)
        process_split(client, split, items, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
