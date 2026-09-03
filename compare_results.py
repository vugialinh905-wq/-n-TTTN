# -*- coding: utf-8 -*-
"""
compare_results.py
Giai đoạn 4 — Gộp kết quả ProtoNet (Người 1) và Fine-tuning baseline
(Người 2) thành 1 bảng so sánh.

KHÔNG cần train lại, KHÔNG cần load model — chỉ đọc lại các file JSON
mà evaluate() ở mỗi file train_*.py đã tự lưu vào Drive từ Giai đoạn 3.

Yêu cầu trước khi chạy: cả 2 người đã chạy evaluate() xong (Người 1 qua
train_protonet.py, Người 2 qua train_baseline_finetune.py BẢN ĐÃ TUNE),
kết quả đã nằm sẵn trong RESULTS_DIR trên Drive.
"""

import os
import json

from episode_utils import RESULTS_DIR


def load_result(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"CẢNH BÁO: chưa thấy file {path} — có thể người tương ứng "
              f"chưa chạy evaluate() cho cấu hình này.")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_comparison_table():
    rows = []
    for k in [1, 5, 10]:
        proto = load_result(f'protonet_{k}shot.json')
        base = load_result(f'baseline_finetune_{k}shot.json')

        for result in (proto, base):
            if result is None:
                continue
            rows.append({
                'method': result['method'],
                'k_shot': result['k_shot'],
                'accuracy': result['mean_accuracy'],
                'ci95': result['ci95'],
                'inference_time_ms': result['mean_inference_time_sec'] * 1000,
                'train_time_sec': result.get('train_time_sec'),  # None với baseline (không có pha train riêng)
            })
    return rows


def print_table(rows):
    header = f"{'Phương pháp':28s} | {'K-shot':6s} | {'Accuracy':17s} | {'Thời gian/episode':18s} | {'Thời gian train':16s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        acc_str = f"{r['accuracy']*100:.2f}% ± {r['ci95']*100:.2f}%"
        time_str = f"{r['inference_time_ms']:.2f} ms"
        train_str = (f"{r['train_time_sec']/60:.1f} phút"
                     if r['train_time_sec'] is not None else "(không có, fine-tune theo episode)")
        print(f"{r['method']:28s} | {r['k_shot']:<6d} | {acc_str:17s} | {time_str:18s} | {train_str:16s}")


if __name__ == '__main__':
    rows = build_comparison_table()
    if not rows:
        print("Chưa có kết quả nào để so sánh — kiểm tra lại cả 2 người "
              "đã chạy evaluate() chưa.")
    else:
        print_table(rows)

        # Lưu lại bảng gộp để dùng vẽ biểu đồ ở Giai đoạn 5
        out_path = os.path.join(RESULTS_DIR, 'comparison_table.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nĐã lưu bảng gộp tại: {out_path}")
