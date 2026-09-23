# -*- coding: utf-8 -*-
"""
build_prototypes.py
"""

import os
import json
import random

import torch

from data_prep import set_seed, SEED, eval_transform, load_and_preprocess
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta
from train_protonet import Conv4Backbone

# Số ảnh tham chiếu mỗi lớp dùng để tính prototype cho demo — nhiều hơn K_SHOT (1/5/10) dùng khi đánh giá khoa học, để prototype ổn định hơn khi minh họa.
N_REFERENCE_PER_CLASS = 20

PROTONET_CKPT = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')
OUT_PATH = os.path.join(RESULTS_DIR, 'prototypes.json')


def pick_reference_images(class_to_images, novel_classes, n_per_class, seed=SEED):
    """Rút cố định (theo seed) n_per_class ảnh mỗi lớp durian — cố định 1 bộ ảnh tham chiếu."""
    rng = random.Random(seed) # Tạo bộ random riêng có seed cố định
    reference = {}
    for c in novel_classes:
        paths = class_to_images[c]
        n = min(n_per_class, len(paths)) # Nếu lớp có ít ảnh hơn yêu cầu thì dùng tất cả ảnh hiện có
        reference[c] = rng.sample(paths, n)
    return reference


def build_protonet_prototypes(reference_images, novel_classes):
    model = Conv4Backbone().to(DEVICE) # Tạo lại kiến trúc ProtoNet
    model.load_state_dict(torch.load(PROTONET_CKPT, map_location=DEVICE))
    model.eval()

    prototypes = {}
    with torch.no_grad(): # Không tính gradient giúp chạy nhanh và tiết kiệm RAM
        for c in novel_classes:
            imgs = torch.stack([
                load_and_preprocess(p, transform=eval_transform)
                for p in reference_images[c]
            ]).to(DEVICE) # Đọc ảnh và biến thành tensor
            emb = model(imgs)                 # (n_ref, 64) Chuyển ảnh thành embedding
            prototypes[c] = emb.mean(dim=0).cpu().tolist() # Tính prototype
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
    } # Dữ liệu sẽ được lưu thành file JSON gồm: danh sách lớp, số ảnh tham chiếu mỗi lớp, prototype của từng lớp
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu prototypes.json tại: {OUT_PATH}")
    print("Copy file này vào cùng thư mục với streamlit_app.py trước khi chạy demo.")


if __name__ == '__main__':
    main()
