import json
import random
import re


def match_objects_to_explanation(qa_data):
    results = []
    used_qids = set()
    for qa in qa_data:
        if "qid" in qa:
            used_qids.add(str(qa["qid"]))
    for qa in qa_data:
        bounding_boxes = qa.get("bounding_boxes", {})
        if any((not box_data.get("boxes") for box_data in bounding_boxes.values())):
            continue
        while True:
            qid = str(random.randint(100000, 999999))
            if qid not in used_qids:
                used_qids.add(qid)
                break
        image = qa["image"]
        question = qa["question"]
        answer = qa["answer"]
        explanation = qa["explanation"]
        object_to_boxes = {}
        for (obj_name, box_data) in bounding_boxes.items():
            boxes = box_data.get("boxes", [])
            if boxes:
                object_to_boxes[obj_name.lower()] = boxes[0]
        sorted_objects = sorted(object_to_boxes.keys(), key=len, reverse=True)
        replacements = []
        for obj_name in sorted_objects:
            box = object_to_boxes[obj_name]
            pattern_exact = "\\b" + re.escape(obj_name) + "\\b"
            matches = list(re.finditer(pattern_exact, explanation, re.IGNORECASE))
            if not matches:
                words = obj_name.split()
                if len(words) > 1:
                    loose_pattern = (
                        "\\b"
                        + "\\b[\\w\\s]{0,20}?\\b".join((re.escape(w) for w in words))
                        + "\\b"
                    )
                    matches = list(
                        re.finditer(loose_pattern, explanation, re.IGNORECASE)
                    )
            if not matches:
                continue
            match = matches[0]
            (start, end) = match.span()
            overlap = any(
                (s <= start < e or s < end <= e for (s, e, _) in replacements)
            )
            if overlap:
                continue
            replacement = f"<ref>{match.group()}</ref><box>{box}</box>"
            replacements.append((start, end, replacement))
        marked_explanation = explanation
        for (start, end, replacement) in sorted(
            replacements, key=lambda x: x[0], reverse=True
        ):
            marked_explanation = (
                marked_explanation[:start] + replacement + marked_explanation[end:]
            )
        result = {
            "qid": qid,
            "image": image,
            "question": question,
            "answer": answer,
            "explanation": marked_explanation,
            "type": qa.get("type", ""),
            "key_objects": qa.get("key_objects", []),
            "bounding_boxes": bounding_boxes,
        }
        results.append(result)
    return results


def main():
    input_file = "../dataset/train/06_evidence_verification.json"
    output_file = "../dataset/train/07_output.json"
    with open(input_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    results = match_objects_to_explanation(qa_data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
