# -*- coding: utf-8 -*-
"""
calibrate_threshold.py (v3 — thêm temperature scaling)

VẤN ĐỀ CỤ THỂ ĐÃ QUAN SÁT: ảnh con mèo (hoàn toàn không liên quan) vẫn ra
93% "Leaf_Blight" — cao hơn NHIỀU so với mọi ngưỡng hợp lý. Đây không còn
là vấn đề "chọn sai ngưỡng" nữa — mà là softmax(-dist) tự nó ĐÃ QUÁ TỰ TIN
(overconfident) với MỌI ảnh, kể cả ảnh hoàn toàn ngẫu nhiên. Không có
ngưỡng nào cứu được khi tín hiệu gốc đã bị "ép" gần 100% một cách vô căn
cứ.

CÁCH SỬA: thêm hệ số TEMPERATURE (T) khi tính softmax:
    probs = softmax(-dist / T)
T càng lớn, phân phối xác suất càng "dịu" (ít cực đoan), giúp phản ánh
đúng hơn mức độ chắc chắn thật của model. T được hiệu chỉnh bằng chính dữ
liệu (không đoán mò): thử nhiều giá trị T, với mỗi T lại tìm ngưỡng tốt
nhất, chọn CẶP (T, ngưỡng) cho balanced accuracy cao nhất.

SCHEMA — prototypes.json sẽ có thêm (cấu trúc lồng nhau):
        "confidence_threshold": {"protonet": <ngưỡng>}
        "confidence_temperature": {"protonet": <T>}

PHẠM VI: đây là bước hiệu chỉnh NGOẠI TUYẾN. Bản ứng dụng hiện tại
(demo/streamlit_app.py) KHÔNG đọc ngưỡng này và không từ chối ảnh ngoài
phân phối (OOD) — tính năng đó đã được đưa ra khỏi phạm vi đề tài. File
được giữ lại để tái lập phần hiệu chỉnh ngưỡng đã mô tả trong báo cáo.

CHẠY TRÊN COLAB, sau khi đã có models/protonet_conv4_best.pt và
results/prototypes.json (từ build_prototypes.py).
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

# Lưới nhiệt độ thử — trải rộng vì không biết trước độ lớn khoảng cách
# Euclidean thực tế trong không gian embedding 64 chiều
TEMPERATURE_CANDIDATES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

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


def compute_raw_dists(model, prototypes, image_path):
    """Chỉ tính khoảng cách thô (chưa qua softmax) — để sau đó thử nhiều
    temperature khác nhau mà không cần forward lại model mỗi lần."""
    img = Image.open(image_path).convert('RGB')
    x = eval_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(x)
        dists = torch.pow(emb - prototypes, 2).sum(dim=1)
    return dists.cpu()


def top1_probs_at_temperature(dists_matrix, temperature):
    """dists_matrix: (N_ảnh, n_lớp). Trả về list top1 probability của
    từng ảnh, ứng với 1 giá trị temperature cụ thể."""
    probs = F.softmax(-dists_matrix / temperature, dim=1)
    return probs.max(dim=1).values.tolist()


def find_best_threshold(positive_scores, negative_scores):
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


def main():
    set_seed(SEED)
    model, proto_data, novel_classes, prototypes = load_protonet_and_prototypes()

    meta = load_meta()
    class_to_images = meta['class_to_images']

    # --- Thu thập khoảng cách thô cho mẫu DƯƠNG (Durian thật) ---
    positive_paths = []
    for c in novel_classes:
        positive_paths += random.sample(
            class_to_images[c], min(N_SAMPLES_PER_CLASS, len(class_to_images[c]))
        )
    positive_dists = torch.stack(
        [compute_raw_dists(model, prototypes, p) for p in positive_paths]
    )

    # --- Thu thập khoảng cách thô cho mẫu ÂM (không phải Durian) ---
    negative_pool = meta['base_classes'] + meta['validation_classes']
    negative_classes = random.sample(negative_pool, min(N_NEGATIVE_CLASSES, len(negative_pool)))
    negative_paths = []
    for c in negative_classes:
        negative_paths += random.sample(
            class_to_images[c], min(N_SAMPLES_PER_CLASS, len(class_to_images[c]))
        )
    negative_dists = torch.stack(
        [compute_raw_dists(model, prototypes, p) for p in negative_paths]
    )

    print(f"Đã tính khoảng cách cho {len(positive_paths)} ảnh Durian thật "
          f"và {len(negative_paths)} ảnh không phải Durian.")
    print("=" * 70)
    print(f"{'Temperature':>12} | {'Ngưỡng tốt nhất':>16} | {'Balanced Acc':>13}")
    print("-" * 70)

    best_temp, best_threshold, best_balanced_acc = 1, 0.5, -1.0
    for T in TEMPERATURE_CANDIDATES:
        pos_scores = top1_probs_at_temperature(positive_dists, T)
        neg_scores = top1_probs_at_temperature(negative_dists, T)
        threshold, balanced_acc = find_best_threshold(pos_scores, neg_scores)
        print(f"{T:>12} | {threshold:>16.3f} | {balanced_acc * 100:>12.1f}%")
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_temp = T
            best_threshold = threshold

    print("=" * 70)
    print(f"CHỌN: temperature={best_temp}, ngưỡng={best_threshold:.3f}, "
          f"balanced accuracy={best_balanced_acc * 100:.1f}%")

    if best_balanced_acc < 0.65:
        print("\nCẢNH BÁO: balanced accuracy vẫn dưới 65% dù đã thử nhiều "
              "temperature — domain shift có thể quá nặng để cơ chế này "
              "tách biệt tốt. Cần ghi rõ hạn chế này trong báo cáo.")

    # --- Kiểm tra cụ thể: với temperature+ngưỡng đã chọn, ảnh mẫu âm có
    #     ảnh nào vẫn bị "overconfident" như trường hợp con mèo 93% không? ---
    pos_scores_best = top1_probs_at_temperature(positive_dists, best_temp)
    neg_scores_best = top1_probs_at_temperature(negative_dists, best_temp)
    max_neg_score = max(neg_scores_best)
    print(f"\nĐộ tự tin CAO NHẤT trong số ảnh KHÔNG phải Durian (sau khi "
          f"áp temperature): {max_neg_score:.3f}")
    if max_neg_score > 0.7:
        print("CẢNH BÁO: vẫn có ít nhất 1 ảnh không phải Durian bị model "
              "quá tự tin (>70%) — hiện tượng overconfident CHƯA hết hoàn "
              "toàn dù đã thêm temperature. Đây là hạn chế nên nêu rõ.")

    # --- Lưu vào prototypes.json — ĐÚNG cấu trúc lồng nhau app đang dùng ---
    proto_data.setdefault('confidence_threshold', {})
    proto_data.setdefault('confidence_temperature', {})
    proto_data['confidence_threshold']['protonet'] = float(best_threshold)
    proto_data['confidence_temperature']['protonet'] = float(best_temp)
    proto_data['confidence_threshold']['protonet_balanced_acc'] = float(best_balanced_acc)

    with open(PROTO_PATH, 'w', encoding='utf-8') as f:
        json.dump(proto_data, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu vào: {PROTO_PATH}")


if __name__ == '__main__':
    main()
