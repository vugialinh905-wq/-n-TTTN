# -*- coding: utf-8 -*-
"""
build_support_set.py
Fine-tuning chạy trực tiếp (live) trong demo Streamlit.
"""

import os
import json
import random
import shutil

from data_prep import set_seed, SEED, DRIVE_BASE_DIR
from episode_utils import load_meta


N_SUPPORT_PER_CLASS = 10

SUPPORT_SET_DIR = os.path.normpath(os.path.join(DRIVE_BASE_DIR, '..', 'support_set')) # Thư mục chứa ảnh support sau khi copy Streamlit sẽ đọc ảnh từ đây
MANIFEST_PATH = os.path.join(SUPPORT_SET_DIR, 'manifest.json')


def pick_support_images(class_to_images, novel_classes, n_per_class, seed=SEED):
    """Rút cố định (theo seed) """
    rng = random.Random(seed)
    chosen = {}
    for c in novel_classes:
        paths = class_to_images[c] # Danh sách toàn bộ ảnh của lớp hiện tại
        n = min(n_per_class, len(paths))
        chosen[c] = rng.sample(paths, n)
    return chosen


def copy_support_set(chosen):
  
    if os.path.exists(SUPPORT_SET_DIR): 
        shutil.rmtree(SUPPORT_SET_DIR) # Xóa thư mục cũ nếu đã tồn tại Tránh bị lẫn ảnh của lần chạy trước
    os.makedirs(SUPPORT_SET_DIR, exist_ok=True) # Tạo thư mục support_set mới

    manifest = {'novel_classes': list(chosen.keys()), 'n_support_per_class': N_SUPPORT_PER_CLASS,
                'classes': {}} # File manifest lưu thông tin: các lớp bệnh, số ảnh support mỗi lớp, đường dẫn ảnh của từng lớp

    for cls, paths in chosen.items(): # Tạo thư mục riêng cho từng lớp
        cls_dir = os.path.join(SUPPORT_SET_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        rel_paths = []
        for i, src in enumerate(paths):
            ext = os.path.splitext(src)[1]
            dst_name = f"{i:02d}{ext}" # đổi tên ảnh
            shutil.copyfile(src, os.path.join(cls_dir, dst_name)) # Copy ảnh gốc sang thư mục support_set
            rel_paths.append(f"{cls}/{dst_name}")
        manifest['classes'][cls] = rel_paths

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2) # Lưu manifest.json, Streamlit sẽ đọc file này, để biết ảnh nào thuộc lớp nào

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
