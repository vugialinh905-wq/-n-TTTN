# -*- coding: utf-8 -*-
"""
build_support_set.py
Giai đoạn 5 (bản demo nâng cấp) — Chuẩn bị bộ ảnh mẫu THẬT cho baseline
Fine-tuning chạy trực tiếp (live) trong demo Streamlit.

VÌ SAO CẦN FILE NÀY (khác với build_prototypes.py):
- ProtoNet: bản chất phương pháp là KHÔNG train lại khi gặp lớp mới — chỉ
  cần 1 vector prototype (trung bình embedding) mỗi lớp. Vì vậy chỉ cần
  lưu SẴN vector đó vào prototypes.json là đủ, không cần giữ lại ảnh gốc.
- Fine-tuning: bản chất phương pháp NGƯỢC LẠI — mỗi lần gặp lớp mới đều
  phải huấn luyện lại 1 lớp Linear MỚI trên đúng K ảnh support của lớp đó
  (xem finetune_on_episode() trong train_baseline_finetune.py). Nếu demo
  chỉ nạp sẵn 1 bộ trọng số Linear đã train từ trước (như bản
  build_prototypes.py cũ), demo sẽ KHÔNG còn đúng bản chất "fine-tuning
  live" nữa — chỉ còn là nhận diện bằng classifier tĩnh, giống ProtoNet
  về cách vận hành, làm mất đi đúng điểm khác biệt cần so sánh.
  => Do đó cần giữ lại ẢNH GỐC (không phải vector) của bộ support set, để
  streamlit_app.py có thể tự fine-tune lại lớp Linear ngay lúc chạy demo,
  y hệt cách train_baseline_finetune.py làm khi đánh giá khoa học.

File này chạy 1 LẦN trong Colab (nơi có class_to_images.json và ảnh gốc),
copy ảnh support set thật ra 1 thư mục nhẹ (vài chục ảnh, không phải cả
dataset), rồi thư mục đó được tải về máy cá nhân cùng với prototypes.json
và protonet_conv4_best.pt.

Yêu cầu chạy trước: data_prep.py, episode_utils.py (đã import được).
"""

import os
import json
import random
import shutil

from data_prep import set_seed, SEED, DRIVE_BASE_DIR
from episode_utils import load_meta

# Số ảnh support mỗi lớp — dùng ĐÚNG K=10 (không phải 20 như
# build_prototypes.py), vì đây là ảnh sẽ được fine-tune LIVE mỗi lần chạy
# demo — cần thời gian fine-tune đủ nhanh, và K=10 là cấu hình có kết quả
# tốt nhất trong 3 cấu hình đã đánh giá chính thức
# (baseline_finetune_10shot.json: accuracy 53,61%).
N_SUPPORT_PER_CLASS = 10

# Nằm NGANG CẤP với models/ và results/ trong project (giống cấu trúc
# demo/ trên máy cá nhân: demo/support_set/, demo/models/, demo/results/)
SUPPORT_SET_DIR = os.path.normpath(os.path.join(DRIVE_BASE_DIR, '..', 'support_set'))
MANIFEST_PATH = os.path.join(SUPPORT_SET_DIR, 'manifest.json')


def pick_support_images(class_to_images, novel_classes, n_per_class, seed=SEED):
    """Rút cố định (theo seed) — CÙNG cách chọn với pick_reference_images()
    trong build_prototypes.py, để việc chọn ảnh nhất quán, có thể tái lập,
    không phải rút ngẫu nhiên mỗi lần chạy lại file này."""
    rng = random.Random(seed)
    chosen = {}
    for c in novel_classes:
        paths = class_to_images[c]
        n = min(n_per_class, len(paths))
        chosen[c] = rng.sample(paths, n)
    return chosen


def copy_support_set(chosen):
    """Copy ảnh gốc ra SUPPORT_SET_DIR, tổ chức theo từng thư mục con =
    tên lớp, giống cấu trúc dataset gốc — dễ mở kiểm tra bằng mắt, và dễ
    đọc lại bằng os.listdir() phía streamlit_app.py."""
    if os.path.exists(SUPPORT_SET_DIR):
        shutil.rmtree(SUPPORT_SET_DIR)
    os.makedirs(SUPPORT_SET_DIR, exist_ok=True)

    manifest = {'novel_classes': list(chosen.keys()), 'n_support_per_class': N_SUPPORT_PER_CLASS,
                'classes': {}}

    for cls, paths in chosen.items():
        cls_dir = os.path.join(SUPPORT_SET_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        rel_paths = []
        for i, src in enumerate(paths):
            ext = os.path.splitext(src)[1]
            dst_name = f"{i:02d}{ext}"
            shutil.copyfile(src, os.path.join(cls_dir, dst_name))
            rel_paths.append(f"{cls}/{dst_name}")
        manifest['classes'][cls] = rel_paths

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main():
    set_seed(SEED)
    meta = load_meta()
    novel_classes = meta['novel_classes']
    class_to_images = meta['class_to_images']

    print(f"Đang chọn {N_SUPPORT_PER_CLASS} ảnh support/lớp cho "
          f"{len(novel_classes)} lớp bệnh sầu riêng...")
    chosen = pick_support_images(class_to_images, novel_classes, N_SUPPORT_PER_CLASS)

    print(f"Đang copy ảnh ra: {SUPPORT_SET_DIR}")
    manifest = copy_support_set(chosen)

    total_images = sum(len(v) for v in manifest['classes'].values())
    print(f"\nXong. Đã copy {total_images} ảnh vào {SUPPORT_SET_DIR}")
    print(f"Đã lưu manifest tại: {MANIFEST_PATH}")
    print("\nBước tiếp theo: tải NGUYÊN thư mục support_set/ này (kèm "
          "manifest.json) về máy cá nhân, đặt cùng cấp với models/ và "
          "results/ trong thư mục demo/, trước khi chạy streamlit_app.py.")


if __name__ == '__main__':
    main()
