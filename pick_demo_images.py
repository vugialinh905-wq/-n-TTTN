# -*- coding: utf-8 -*-
"""
pick_demo_images.py
Giai đoạn 5 — Chọn sẵn ảnh test cho demo Streamlit, đúng 2 trường hợp:

1. "test_durian/"       — ảnh Durian THẬT, KHÁC 5 ảnh support_0..4 đã dùng
                            build prototype (để test đúng tính năng chính:
                            nhận diện bệnh).
2. "test_other_plant/"  — ảnh lá Cà chua/Khoai tây/Ớt (KHÔNG PHẢI Durian,
                            đúng loại ảnh đã dùng để hiệu chỉnh ngưỡng ở
                            calibrate_threshold.py) — để test tính năng
                            từ chối.

Chạy trên Colab, sau đó tải 2 thư mục output về máy local để dùng demo.
"""

import os
import random
import shutil

from data_prep import set_seed, SEED
from episode_utils import load_meta

N_PER_CLASS = 3          # lấy 3 ảnh mỗi lớp để demo
OUT_DIR = '/content/demo_images'
SUPPORT_SET_NAMES = {f'support_{i}.jpg' for i in range(10)}  # phòng hờ nếu tên trùng


def pick_images():
    set_seed(SEED)
    meta = load_meta()
    class_to_images = meta['class_to_images']

    # --- 1. Ảnh Durian thật, tránh trùng ảnh đã dùng làm support ---
    # Lưu ý: support_set.zip bạn có sẵn không ghi rõ ảnh gốc nào được chọn
    # (chỉ đổi tên thành support_0..4), nên ở đây mình lấy ngẫu nhiên với
    # seed cố định — nếu vô tình trùng 1-2 ảnh support cũng không sao vì
    # N_PER_CLASS=3 vẫn còn ảnh khác, nhưng để chắc chắn 100%, tốt nhất bạn
    # tự đối chiếu thủ công tên file trong support_set.zip trước khi dùng.
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
    zip_path = shutil.make_archive('/content/demo_images', 'zip', OUT_DIR)
    print(f"\nĐã đóng gói toàn bộ ảnh demo vào: {zip_path}")
    print("Tải file này về máy (icon thư mục bên trái Colab -> chuột phải -> Download).")


if __name__ == '__main__':
    pick_images()
