# -*- coding: utf-8 -*-
"""
build_prototypes.py
Giai đoạn 5 — Chuẩn bị "bộ não" cho phía ProtoNet của demo Streamlit.

Vì sao chỉ có ProtoNet ở đây (không có Fine-tuning nữa):
Bản chất Prototypical Network là KHÔNG train lại khi gặp lớp mới — chỉ
cần 1 vector prototype (trung bình embedding) cố định mỗi lớp là đủ để so
khớp ảnh mới. Vì vậy có thể "đóng băng" phần này thành prototypes.json,
tính 1 lần duy nhất, dùng lại mãi.

Phần Fine-tuning KHÔNG làm theo cách này nữa (xem build_support_set.py):
đúng bản chất phương pháp, Fine-tuning phải huấn luyện lại lớp Linear mỗi
lần gặp lớp mới bằng ảnh support thật — nên streamlit_app.py sẽ tự
fine-tune LIVE bằng ảnh trong support_set/, không nạp trọng số có sẵn.

Yêu cầu chạy trước file này (đúng thứ tự):
1. data_prep.py     — đã tạo xong class_to_images.json
2. episode_utils.py — nằm cùng thư mục, import được
3. train_protonet.py đã chạy xong, có models/protonet_conv4_best.pt
"""

import os
import json
import random

import torch

from data_prep import set_seed, SEED, eval_transform, load_and_preprocess
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta
from train_protonet import Conv4Backbone

# Số ảnh tham chiếu mỗi lớp dùng để tính prototype cho demo — nhiều hơn
# K_SHOT (1/5/10) dùng khi đánh giá khoa học, để prototype ổn định hơn
# khi minh họa. Không ảnh hưởng đến accuracy chính thức (vẫn lấy từ
# results_snapshot/*.json ở Giai đoạn 3-4).
N_REFERENCE_PER_CLASS = 20

PROTONET_CKPT = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')
OUT_PATH = os.path.join(RESULTS_DIR, 'prototypes.json')


def pick_reference_images(class_to_images, novel_classes, n_per_class, seed=SEED):
    """Rút cố định (theo seed) n_per_class ảnh mỗi lớp durian — cố định
    1 lần, để mọi lần chạy demo đều dùng đúng cùng 1 bộ ảnh tham chiếu."""
    rng = random.Random(seed)
    reference = {}
    for c in novel_classes:
        paths = class_to_images[c]
        n = min(n_per_class, len(paths))
        reference[c] = rng.sample(paths, n)
    return reference


def build_protonet_prototypes(reference_images, novel_classes):
    model = Conv4Backbone().to(DEVICE)
    model.load_state_dict(torch.load(PROTONET_CKPT, map_location=DEVICE))
    model.eval()

    prototypes = {}
    with torch.no_grad():
        for c in novel_classes:
            imgs = torch.stack([
                load_and_preprocess(p, transform=eval_transform)
                for p in reference_images[c]
            ]).to(DEVICE)
            emb = model(imgs)                 # (n_ref, 64)
            prototypes[c] = emb.mean(dim=0).cpu().tolist()
    return prototypes


def main():
    set_seed(SEED)
    meta = load_meta()
    novel_classes = meta['novel_classes']
    class_to_images = meta['class_to_images']

    print(f"Đang chọn {N_REFERENCE_PER_CLASS} ảnh tham chiếu/lớp cho "
          f"{len(novel_classes)} lớp bệnh sầu riêng...")
    reference_images = pick_reference_images(
        class_to_images, novel_classes, N_REFERENCE_PER_CLASS
    )

    print("Đang tính prototype cho Prototypical Network...")
    protonet_prototypes = build_protonet_prototypes(reference_images, novel_classes)

    output = {
        'novel_classes': novel_classes,
        'n_reference_per_class': N_REFERENCE_PER_CLASS,
        'protonet_prototypes': protonet_prototypes,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu prototypes.json tại: {OUT_PATH}")
    print("Copy file này vào cùng thư mục với streamlit_app.py trước khi chạy demo.")


if __name__ == '__main__':
    main()
