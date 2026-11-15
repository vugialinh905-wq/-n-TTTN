# -*- coding: utf-8 -*-
"""
pick_demo_images.py
Giai đoạn 5 — Chọn sẵn ảnh thử nhanh cho demo Streamlit, gồm 2 nhóm:

1. "test_durian/"       — ảnh Durian thật, để thử tính năng chính của
                            ứng dụng (nhận diện 6 lớp lá/bệnh lá sầu riêng).
2. "test_other_plant/"  — ảnh lá Cà chua/Khoai tây/Ớt (KHÔNG phải Durian,
                            cùng loại ảnh dùng làm mẫu âm ở
                            calibrate_threshold.py) — để minh hoạ giới hạn
                            của ứng dụng với ảnh ngoài 6 lớp.

Lưu ý: ảnh Durian được rút ngẫu nhiên theo seed cố định từ toàn bộ ảnh của
mỗi lớp, KHÔNG loại trừ các ảnh đã nằm trong support_set/ hay đã dùng để
tính prototype. Vì vậy 2 thư mục này chỉ dùng để thao tác thử ứng dụng,
không dùng làm số liệu đánh giá.

Chạy trên Colab, sau đó tải 2 thư mục output về máy local để dùng demo.
"""

import os
import random
import shutil

from data_prep import set_seed, SEED, LOCAL_BASE_DIR
from episode_utils import load_meta

N_PER_CLASS = 3          # lấy 3 ảnh mỗi lớp để demo
OUT_DIR = os.path.join(os.path.dirname(LOCAL_BASE_DIR), 'demo_images')  # Colab: /content/demo_images


def pick_images():
    set_seed(SEED)
    meta = load_meta()
    class_to_images = meta['class_to_images']

    # --- 1. Ảnh Durian thật (rút ngẫu nhiên theo seed, xem lưu ý đầu file) ---
    durian_dir = os.path.join(OUT_DIR, 'test_durian')
    os.makedirs(durian_dir, exist_ok=True)
    for c in meta['novel_classes']:
        class_out = os.path.join(durian_dir, c)
        os.makedirs(class_out, exist_ok=True)
        paths = random.sample(class_to_images[c], min(N_PER_CLASS, len(class_to_images[c])))
        for i, p in enumerate(paths):
            shutil.copy(p, os.path.join(class_out, f'test_{i}.jpg'))
        print(f"[Durian] {c}: đã copy {len(paths)} ảnh vào {class_out}")

    # --- 2. Ảnh lá cây KHÁC (Cà chua/Khoai tây/Ớt) ---
    other_dir = os.path.join(OUT_DIR, 'test_other_plant')
    os.makedirs(other_dir, exist_ok=True)
    negative_pool = meta['base_classes'] + meta['validation_classes']
    chosen_classes = random.sample(negative_pool, min(5, len(negative_pool)))
    for c in chosen_classes:
        class_out = os.path.join(other_dir, c)
        os.makedirs(class_out, exist_ok=True)
        paths = random.sample(class_to_images[c], min(N_PER_CLASS, len(class_to_images[c])))
        for i, p in enumerate(paths):
            shutil.copy(p, os.path.join(class_out, f'test_{i}.jpg'))
        print(f"[Lá cây khác] {c}: đã copy {len(paths)} ảnh vào {class_out}")

    # --- Nén lại thành 1 file zip để tải về máy local cho tiện ---
    zip_path = shutil.make_archive(OUT_DIR, 'zip', OUT_DIR)
    print(f"\nĐã đóng gói toàn bộ ảnh demo vào: {zip_path}")
    print("Tải file này về máy (icon thư mục bên trái Colab -> chuột phải -> Download).")


if __name__ == '__main__':
    pick_images()
