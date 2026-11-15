# -*- coding: utf-8 -*-
"""
multiseed_protonet.py
Huấn luyện ProtoNet với nhiều seed, báo trung bình ± độ lệch chuẩn giữa các
lần huấn luyện. Chạy lại được sau khi Colab ngắt: seed nào xong rồi thì bỏ qua.
Mọi file lưu kèm tag (seed1, seed2...), không đụng vào checkpoint chuẩn.

    import multiseed_protonet as ms
    ms.run_all()                                          # 5 seed, cách đã sửa
    ms.run_all(seeds=[1, 2, 3], isolate_val_rng=False)    # tuỳ chọn: cách cũ, để đo ảnh hưởng
    ms.summarize()                                        # xem lại bảng tổng hợp bất cứ lúc nào
"""

import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR
from train_protonet import Conv4Backbone, train, evaluate

SEEDS = [1, 2, 3, 4, 5]
K_LIST = [1, 5, 10]


def _eval_files(tag):
    return [os.path.join(RESULTS_DIR, f'protonet_{k}shot_{tag}.json') for k in K_LIST]


def run_seed(seed, prefix='seed', isolate_val_rng=True):
    """Train + đánh giá 1 seed. Bỏ qua nếu đã có kết quả; chỉ đánh giá lại nếu đã train xong."""
    tag = f'{prefix}{seed}'
    if all(os.path.exists(p) for p in _eval_files(tag)):
        print(f"[{tag}] đã có kết quả, bỏ qua.")
        return

    best = os.path.join(MODELS_DIR, f'protonet_conv4_best_{tag}.pt')
    summary = os.path.join(RESULTS_DIR, f'protonet_training_summary_{tag}.json')
    if os.path.exists(best) and os.path.exists(summary):
        print(f"[{tag}] đã train xong, chỉ đánh giá lại.")
        model = Conv4Backbone().to(DEVICE)
        model.load_state_dict(torch.load(best, map_location=DEVICE))
    else:
        print(f"[{tag}] bắt đầu huấn luyện (isolate_val_rng={isolate_val_rng})...")
        # overwrite=True chỉ ghi đè file của đúng tag này (lần train dở), không đụng file chuẩn
        model = train(seed=seed, tag=tag, isolate_val_rng=isolate_val_rng, overwrite=True)
    evaluate(model, tag=tag)


def run_all(seeds=SEEDS, isolate_val_rng=True):
    prefix = 'seed' if isolate_val_rng else 'legacy'
    for s in seeds:
        run_seed(s, prefix, isolate_val_rng)
    summarize(prefix, seeds)


def _load_acc(prefix, seeds):
    """Accuracy (%) ProtoNet theo từng K, mỗi seed một giá trị."""
    acc = {k: [] for k in K_LIST}
    for s in seeds:
        for k in K_LIST:
            p = os.path.join(RESULTS_DIR, f'protonet_{k}shot_{prefix}{s}.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as f:
                    acc[k].append(json.load(f)['mean_accuracy'] * 100)
    return acc


def _load_finetune():
    """Fine-tuning: (accuracy %, CI95 %) theo từng K, từ file kết quả chuẩn."""
    ft = {}
    for k in K_LIST:
        p = os.path.join(RESULTS_DIR, f'baseline_finetune_{k}shot.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            ft[k] = (d['mean_accuracy'] * 100, d['ci95'] * 100)
    return ft


def plot_summary(acc, ft, prefix, out_path):
    """Cột = trung bình ProtoNet (thanh lỗi = độ lệch chuẩn giữa seed), chấm = từng seed."""
    x = np.arange(len(K_LIST))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)

    means = [np.mean(acc[k]) if acc[k] else np.nan for k in K_LIST]
    stds = [np.std(acc[k], ddof=1) if len(acc[k]) > 1 else 0 for k in K_LIST]
    ax.bar(x - w / 2, means, w, yerr=stds, capsize=4, color='#1F4E78',
           label='Prototypical Network (TB ± độ lệch chuẩn giữa các seed)')
    for i, k in enumerate(K_LIST):
        n = len(acc[k])
        if n:
            jitter = np.linspace(-0.07, 0.07, n) if n > 1 else [0]
            ax.scatter(x[i] - w / 2 + np.array(jitter), acc[k], s=22,
                       color='white', edgecolor='black', zorder=3)

    ft_m = [ft[k][0] if k in ft else np.nan for k in K_LIST]
    ft_c = [ft[k][1] if k in ft else 0 for k in K_LIST]
    ax.bar(x + w / 2, ft_m, w, yerr=ft_c, capsize=4, color='#C0622A',
           label='Fine-tuning (± CI95 theo episode)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}-shot' for k in K_LIST])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 65)
    ax.axhline(20, color='#999999', linestyle='--', linewidth=1)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def summarize(prefix='seed', seeds=SEEDS):
    acc = _load_acc(prefix, seeds)
    ft = _load_finetune()

    print(f"\nProtoNet ({prefix}): trung bình ± độ lệch chuẩn GIỮA CÁC LẦN HUẤN LUYỆN.")
    print("Fine-tuning: 1 lần chạy ± CI95 do lấy mẫu episode (hai đại lượng này khác nhau).\n")

    stats = {}
    for k in K_LIST:
        a = np.array(acc[k])
        if len(a) == 0:
            continue
        std = float(a.std(ddof=1)) if len(a) > 1 else float('nan')
        stats[k] = {'mean': float(a.mean()), 'std': std, 'min': float(a.min()),
                    'max': float(a.max()), 'n': int(len(a)), 'values': [float(v) for v in a]}

        line = (f"{k:>2}-shot | ProtoNet {a.mean():5.2f} ± {std:4.2f} "
                f"(n={len(a)}, từ {a.min():5.2f} đến {a.max():5.2f})")
        if k in ft:
            inside = a.min() <= ft[k][0] <= a.max()
            line += (f" | Fine-tuning {ft[k][0]:5.2f} ± {ft[k][1]:4.2f}"
                     f" | nằm trong khoảng ProtoNet: {'có' if inside else 'không'}")
        print(line)
        print("          từng seed:", ", ".join(f"{v:.2f}" for v in a))

    if not stats:
        print("Chưa có kết quả nào.")
        return

    out_json = os.path.join(RESULTS_DIR, f'multiseed_summary_{prefix}.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'prefix': prefix, 'protonet': stats,
                   'finetune': {k: {'mean': v[0], 'ci95': v[1]} for k, v in ft.items()}},
                  f, ensure_ascii=False, indent=2)
    out_png = os.path.join(RESULTS_DIR, f'multiseed_comparison_{prefix}.png')
    plot_summary(acc, ft, prefix, out_png)
    print(f"\nĐã lưu: {out_json}\nĐã lưu: {out_png}")
