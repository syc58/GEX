import json
from tqdm import tqdm
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import colorsys

AREA_RATIO_THRESHOLD = 0.6
TOPK_BOXES = 3


def box_area(box):
    (x0, y0, x1, y1) = box
    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))


def select_topk_and_union(boxes, scores, topk=3):
    if len(boxes) == 0:
        return (boxes, scores)
    k = min(topk, len(boxes))
    topk_indices = torch.argsort(scores, descending=True)[:k]
    topk_boxes = boxes[topk_indices]
    topk_scores = scores[topk_indices]
    x0 = topk_boxes[:, 0].min()
    y0 = topk_boxes[:, 1].min()
    x1 = topk_boxes[:, 2].max()
    y1 = topk_boxes[:, 3].max()
    if x1 <= x0 or y1 <= y0:
        best_idx = scores.argmax()
        return (boxes[best_idx : best_idx + 1], scores[best_idx : best_idx + 1])
    union_box = torch.tensor([[x0, y0, x1, y1]], device=boxes.device, dtype=boxes.dtype)
    union_score = topk_scores.mean().unsqueeze(0)
    return (union_box, union_score)


def should_drop_sample(
    final_boxes_dict, image_size, area_ratio_threshold=AREA_RATIO_THRESHOLD
):
    (img_w, img_h) = image_size
    img_area = float(img_w * img_h)
    large_box_count = 0
    for (_, bbox_data) in final_boxes_dict.items():
        if not bbox_data["boxes"]:
            continue
        box = bbox_data["boxes"][0]
        ratio = box_area(box) / img_area if img_area > 0 else 0.0
        if ratio >= area_ratio_threshold:
            large_box_count += 1
    if large_box_count >= 2:
        return (
            True,
            f"drop: {large_box_count} objects have area ratio >= {area_ratio_threshold}",
        )
    return (False, "")


def run_grounding(
    image,
    text_labels,
    processor,
    model,
    threshold=0.3,
    text_threshold=0.3,
    device="cpu",
):
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )
    return results[0]


def generate_colors(n):
    if n <= 0:
        return []
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(tuple((int(c * 255) for c in rgb)))
    return colors


def clamp_and_sort_box(box, img_w, img_h):
    (x0, y0, x1, y1) = box
    x0 = max(0, min(float(x0), img_w - 1))
    x1 = max(0, min(float(x1), img_w - 1))
    y0 = max(0, min(float(y0), img_h - 1))
    y1 = max(0, min(float(y1), img_h - 1))
    (x0, x1) = (min(x0, x1), max(x0, x1))
    (y0, y1) = (min(y0, y1), max(y0, y1))
    return [int(x0), int(y0), int(x1), int(y1)]


def visualize_results(image, result, output_path):
    if len(result["boxes"]) == 0:
        image.save(output_path)
        return
    draw = ImageDraw.Draw(image)
    (img_w, img_h) = image.size
    BOX_WIDTH = 6
    FONT_SIZE = 28
    TEXT_PADDING = 6
    CORNER_MARGIN = 2
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    unique_labels = list(dict.fromkeys(result["text_labels"]))
    colors = generate_colors(len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    placed_text_boxes = []

    def rect_intersection_area(a, b):
        (ax0, ay0, ax1, ay1) = a
        (bx0, by0, bx1, by1) = b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0
        return (inter_x1 - inter_x0) * (inter_y1 - inter_y0)

    def clamp_rect(rect, img_w, img_h):
        (x0, y0, x1, y1) = rect
        x0 = max(0, min(int(x0), img_w - 1))
        y0 = max(0, min(int(y0), img_h - 1))
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        (x0, x1) = (min(x0, x1), max(x0, x1))
        (y0, y1) = (min(y0, y1), max(y0, y1))
        return [x0, y0, x1, y1]

    def candidate_text_positions(x0, y0, x1, y1, text_w, text_h):
        box_w = text_w + 2 * TEXT_PADDING
        box_h = text_h + 2 * TEXT_PADDING
        candidates = []
        candidates.append(
            (
                "top_left",
                [x0, y0 - box_h - CORNER_MARGIN, x0 + box_w, y0 - CORNER_MARGIN],
            )
        )
        candidates.append(
            (
                "top_right",
                [x1 - box_w, y0 - box_h - CORNER_MARGIN, x1, y0 - CORNER_MARGIN],
            )
        )
        candidates.append(
            (
                "bottom_left",
                [x0, y1 + CORNER_MARGIN, x0 + box_w, y1 + CORNER_MARGIN + box_h],
            )
        )
        candidates.append(
            (
                "bottom_right",
                [x1 - box_w, y1 + CORNER_MARGIN, x1, y1 + CORNER_MARGIN + box_h],
            )
        )
        return candidates

    def choose_best_text_rect(candidates, placed_text_boxes, img_w, img_h):
        best_rect = None
        best_score = None
        for (pos_name, rect) in candidates:
            (raw_x0, raw_y0, raw_x1, raw_y1) = rect
            clamped = clamp_rect(rect, img_w, img_h)
            (cx0, cy0, cx1, cy1) = clamped
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            overflow_penalty = 0
            overflow_penalty += max(0, -raw_x0)
            overflow_penalty += max(0, -raw_y0)
            overflow_penalty += max(0, raw_x1 - (img_w - 1))
            overflow_penalty += max(0, raw_y1 - (img_h - 1))
            overlap_area = 0
            for old_rect in placed_text_boxes:
                overlap_area += rect_intersection_area(clamped, old_rect)
            score = (overlap_area, overflow_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_rect = clamped
        return best_rect

    for (box, score, label) in zip(
        result["boxes"], result["scores"], result["text_labels"]
    ):
        box = box.tolist()
        color = label_to_color[label]
        (x0, y0, x1, y1) = clamp_and_sort_box(box, img_w, img_h)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=color, width=BOX_WIDTH)
        text = f"{label}: {float(score):.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        candidates = candidate_text_positions(x0, y0, x1, y1, text_w, text_h)
        bg_rect = choose_best_text_rect(candidates, placed_text_boxes, img_w, img_h)
        if bg_rect is None:
            fallback = [
                x0,
                y0,
                x0 + text_w + 2 * TEXT_PADDING,
                y0 + text_h + 2 * TEXT_PADDING,
            ]
            bg_rect = clamp_rect(fallback, img_w, img_h)
        (bg_x0, bg_y0, bg_x1, bg_y1) = bg_rect
        if bg_x1 > bg_x0 and bg_y1 > bg_y0:
            draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=color)
            draw.text(
                (bg_x0 + TEXT_PADDING, bg_y0 + TEXT_PADDING),
                text,
                fill="white",
                font=font,
            )
            placed_text_boxes.append(bg_rect)
    image.save(output_path)


def save_json_atomic(data, output_file):
    tmp_file = output_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, output_file)


def load_or_init_output(input_file, output_file):
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    save_json_atomic(data, output_file)
    return data


def is_item_processed(item):
    return "bounding_boxes" in item


def process_data():
    splits = ["train"]
    base_dir = "../dataset"
    image_root = "../DOCCI/images"
    box_images_dir = "../dataset/box_images"
    os.makedirs(box_images_dir, exist_ok=True)
    model_path = "/no_think/shiyecheng/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, local_files_only=True
        ).to(device)
        model.eval()
    except Exception as e:
        return
    threshold = 0.3
    text_threshold = 0.3
    for split in splits:
        input_file = os.path.join(base_dir, split, "04_MC_Check.json")
        output_file = os.path.join(base_dir, split, "05_evidence_localization.json")
        if not os.path.exists(input_file):
            continue
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_items = len(data)
        per_item_vis_index = []
        tmp_counter = {}
        for item in data:
            image_name = item.get("image", "")
            image_id = os.path.splitext(image_name)[0]
            cur_idx = tmp_counter.get(image_id, 0)
            per_item_vis_index.append(cur_idx)
            tmp_counter[image_id] = cur_idx + 1
        items_with_detections = 0
        total_objects = 0
        detected_objects = 0
        dropped_items = 0
        skipped_items = 0
        new_data = []
        for (item_idx, item) in enumerate(
            tqdm(data, desc=f"Detecting objects in {split}")
        ):
            image_name = item.get("image", "")
            extracted_objects = item.get("key_objects", [])
            image_path = os.path.join(image_root, image_name)
            image_id = os.path.splitext(image_name)[0]
            current_image_num = per_item_vis_index[item_idx]
            if is_item_processed(item):
                skipped_items += 1
                new_data.append(item)
                save_json_atomic(new_data, output_file)
                continue
            if not os.path.exists(image_path):
                new_data.append(item)
                save_json_atomic(new_data, output_file)
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                new_data.append(item)
                save_json_atomic(new_data, output_file)
                continue
            bounding_boxes = {}
            total_objects += len(extracted_objects)
            for obj_label in extracted_objects:
                text_labels = [[obj_label]]
                try:
                    result = run_grounding(
                        image=image,
                        text_labels=text_labels,
                        processor=processor,
                        model=model,
                        threshold=threshold,
                        text_threshold=text_threshold,
                        device=device,
                    )
                except Exception as e:
                    bounding_boxes[obj_label] = {"boxes": [], "scores": []}
                    continue
                if len(result["boxes"]) > 0:
                    boxes = result["boxes"]
                    scores = result["scores"]
                    (boxes, scores) = select_topk_and_union(
                        boxes, scores, topk=TOPK_BOXES
                    )
                    boxes_list = []
                    scores_list = []
                    for (box, score) in zip(boxes, scores):
                        box_list = [int(round(float(v), 3)) for v in box.tolist()]
                        score_val = round(float(score.item()), 3)
                        boxes_list.append(box_list)
                        scores_list.append(score_val)
                    bounding_boxes[obj_label] = {
                        "boxes": boxes_list,
                        "scores": scores_list,
                    }
                    if len(boxes_list) > 0:
                        detected_objects += len(boxes_list)
                else:
                    bounding_boxes[obj_label] = {"boxes": [], "scores": []}
            (drop_sample, drop_reason) = should_drop_sample(
                bounding_boxes, image.size, area_ratio_threshold=AREA_RATIO_THRESHOLD
            )
            if drop_sample:
                dropped_items += 1
                save_json_atomic(new_data, output_file)
                continue
            item["bounding_boxes"] = bounding_boxes
            new_data.append(item)
            save_json_atomic(new_data, output_file)
            vis_boxes = []
            vis_scores = []
            vis_labels = []
            for (label, bbox_data) in bounding_boxes.items():
                if bbox_data["boxes"]:
                    for (box, score) in zip(bbox_data["boxes"], bbox_data["scores"]):
                        vis_boxes.append(torch.tensor(box))
                        vis_scores.append(torch.tensor(score))
                        vis_labels.append(label)
            vis_filename = f"{image_id}_{current_image_num}_with_boxes.jpg"
            vis_output_path = os.path.join(box_images_dir, split, vis_filename)
            try:
                vis_image = Image.open(image_path).convert("RGB")
                if vis_boxes:
                    vis_result = {
                        "boxes": torch.stack(vis_boxes),
                        "scores": torch.stack(vis_scores),
                        "text_labels": vis_labels,
                    }
                    visualize_results(vis_image, vis_result, vis_output_path)
                    items_with_detections += 1
                else:
                    vis_image.save(vis_output_path)
            except Exception as e:
                pass
        if total_objects > 0:
            pass


if __name__ == "__main__":
    process_data()
