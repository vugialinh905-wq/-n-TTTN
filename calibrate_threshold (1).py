# -*- coding: utf-8 -*-
"""
calibrate_threshold.py (v2)
Giai đoạn 5 — Hiệu chỉnh ngưỡng từ chối cho ProtoNet, khớp ĐÚNG schema mới
của build_prototypes.py (khóa 'novel_classes' + 'protonet_prototypes',
KHÔNG còn 'class_names'/'prototypes' như bản v1 cũ).

THAY ĐỔI QUAN TRỌNG so với bản v1:
- Tín hiệu độ tự tin đổi từ "xác suất lớp cao nhất" (top1 prob) sang
  "margin" = xác suất lớp #1 trừ lớp #2. Lý do: khi accuracy gốc thấp
  (domain shift nặng, PlantVillage -> Durian), top1 prob bị "phẳng" ra
  gần 1/n_classes cho MỌI ảnh — kể cả ảnh Durian thật lẫn ảnh không phải
  Durian — khiến 2 phân bố chồng lấn nặng, ngưỡng không tách được (đã
  quan sát thực tế: ảnh Durian thật 25%, ảnh con chó 20% — gần như
  không phân biệt được). Margin đo mức độ "quyết đoán" của model, có xu
  hướng tách biệt tốt hơn ngay cả khi độ tự tin tuyệt đối thấp.
- Ngưỡng được chọn bằng grid search tối ưu balanced accuracy (thay vì
  percentile đơn giản như bản v1) — tìm điểm cắt cân bằng tốt nhất giữa
  giữ đúng ảnh Durian thật và loại đúng ảnh không phải Durian.

CHẠY TRÊN COLAB, SAU KHI đã có:
- models/protonet_conv4_best.pt
- results/prototypes.json (từ build_prototypes.py)
"""

import os
import json
import random

import torch
import torch.nn.functional as F
from PIL import Image

from data_prep import set_seed, SEED, eval_transform
from episode_utils import load_meta, DEVICE, MODELS_DIR, RESULTS_DIR
from train_protonet import Conv4Backbone

N_SAMPLES_PER_CLASS = 30
N_NEGATIVE_CLASSES = 10

PROTONET_CKPT = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')
PROTO_PATH = os.path.join(RESULTS_DIR, 'prototypes.json')


def load_protonet_and_prototypes():
    model = Conv4Backbone().to(DEVICE)
    model.load_state_dict(torch.load(PROTONET_CKPT, map_location=DEVICE))
    model.eval()

    with open(PROTO_PATH, 'r', encoding='utf-8') as f:
        proto_data = json.load(f)

    novel_classes = proto_data['novel_classes']
    prototypes = torch.tensor(
        [proto_data['protonet_prototypes'][c] for c in novel_classes]
    ).to(DEVICE)

    return model, proto_data, novel_classes, prototypes


def compute_signals(model, prototypes, image_path):
    """Trả về (top1_prob, margin) cho 1 ảnh — margin = xác suất lớp #1
    trừ lớp #2, tín hiệu chính dùng để hiệu chỉnh ngưỡng."""
    img = Image.open(image_path).convert('RGB')
    x = eval_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(x)
        dists = torch.pow(emb - prototypes, 2).sum(dim=1)
        probs = F.softmax(-dists, dim=0)

    sorted_probs, _ = torch.sort(probs, descending=True)
    top1 = sorted_probs[0].item()
    top2 = sorted_probs[1].item()
    margin = top1 - top2
    return top1, margin


def find_best_threshold(positive_scores, negative_scores):
    """Grid search tìm ngưỡng tối ưu balanced accuracy — cân bằng giữa
    'giữ đúng ảnh Durian thật' và 'loại đúng ảnh không phải Durian',
    thay vì chỉ lấy percentile đơn giản (dễ lệch khi 2 phân bố gần nhau)."""
    candidates = sorted(set(positive_scores + negative_scores))
    best_t, best_balanced_acc = candidates[0], -1.0
    n_pos, n_neg = len(positive_scores), len(negative_scores)

    for t in candidates:
        tp = sum(1 for s in positive_scores if s >= t)
        tn = sum(1 for s in negative_scores if s < t)
        balanced_acc = 0.5 * (tp / n_pos + tn / n_neg)
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_t = t

    return best_t, best_balanced_acc


def summarize(name, scores):
    scores = sorted(scores)
    n = len(scores)
    print(f"{name} ({n} ảnh):")
    print(f"  Nhỏ nhất: {scores[0]:.3f} | Trung vị: {scores[n // 2]:.3f} "
          f"| Lớn nhất: {scores[-1]:.3f}")


def main():
    set_seed(SEED)
    model, proto_data, novel_classes, prototypes = load_protonet_and_prototypes()

    meta = load_meta()
    class_to_images = meta['class_to_images']

    positive_top1, positive_margin = [], []
    for c in novel_classes:
        paths = random.sample(
            class_to_images[c], min(N_SAMPLES_PER_CLASS, len(class_to_images[c]))
        )
        for p in paths:
            t1, m = compute_signals(model, prototypes, p)
            positive_top1.append(t1)
            positive_margin.append(m)

    negative_pool = meta['base_classes'] + meta['validation_classes']
    negative_classes = random.sample(negative_pool, min(N_NEGATIVE_CLASSES, len(negative_pool)))
    negative_top1, negative_margin = [], []
    for c in negative_classes:
        paths = random.sample(
            class_to_images[c], min(N_SAMPLES_PER_CLASS, len(class_to_images[c]))
        )
        for p in paths:
            t1, m = compute_signals(model, prototypes, p)
            negative_top1.append(t1)
            negative_margin.append(m)

    print("=" * 60)
    print("SO SÁNH 2 TÍN HIỆU — top1 prob (cũ) vs margin (mới)")
    print("=" * 60)
    summarize("Top1 prob — ảnh Durian thật", positive_top1)
    summarize("Top1 prob — ảnh KHÔNG phải Durian", negative_top1)
    print()
    summarize("Margin — ảnh Durian thật", positive_margin)
    summarize("Margin — ảnh KHÔNG phải Durian", negative_margin)

    t1_threshold, t1_balanced_acc = find_best_threshold(positive_top1, negative_top1)
    margin_threshold, margin_balanced_acc = find_best_threshold(positive_margin, negative_margin)

    print("\n" + "=" * 60)
    print(f"Balanced accuracy với top1 prob : {t1_balanced_acc * 100:.1f}% "
          f"(ngưỡng {t1_threshold:.3f})")
    print(f"Balanced accuracy với margin     : {margin_balanced_acc * 100:.1f}% "
          f"(ngưỡng {margin_threshold:.3f})")

    if margin_balanced_acc <= t1_balanced_acc:
        print("\nLƯU Ý: margin không tốt hơn top1 prob trong lần hiệu chỉnh "
              "này — có thể domain shift quá nặng khiến CẢ HAI tín hiệu đều "
              "không tách biệt tốt. Vẫn dùng margin (ổn định hơn về lý "
              "thuyết), nhưng cần ghi rõ hạn chế này trong báo cáo.")

    # Lưu cả 2 ngưỡng để streamlit_app.py linh hoạt chọn, ghi rõ balanced
    # accuracy đạt được để biết mức độ tin cậy của cơ chế từ chối này
    proto_data['confidence_threshold_margin'] = float(margin_threshold)
    proto_data['confidence_threshold_margin_balanced_acc'] = float(margin_balanced_acc)
    proto_data['confidence_threshold_top1'] = float(t1_threshold)
    proto_data['confidence_threshold_top1_balanced_acc'] = float(t1_balanced_acc)

    with open(PROTO_PATH, 'w', encoding='utf-8') as f:
        json.dump(proto_data, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu ngưỡng vào: {PROTO_PATH}")
    print("Tải lại prototypes.json này về máy local, dùng cho streamlit_app.py.")


if __name__ == '__main__':
    main()
