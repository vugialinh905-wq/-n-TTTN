# -*- coding: utf-8 -*-
"""
confusion_analysis.py
Dựng ma trận nhầm lẫn (confusion matrix) trên 6 lớp Durian, cho cả 2
phương pháp — để biết chính xác lớp nào hay bị nhầm với lớp nào, thay vì
chỉ nhìn 1 con số accuracy tổng.

Chạy trên Colab, sau khi đã có models/protonet_conv4_best.pt và
results/baseline_tuning_summary.json (để lấy đúng ft_steps/ft_lr đã tune).
"""

import os
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from data_prep import set_seed, SEED, N_WAY
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta, sample_episode, eval_transform
from train_protonet import Conv4Backbone, protonet_forward
from train_baseline_finetune import build_pretrained_backbone, finetune_on_episode

N_EPISODES = 600
K_SHOT_TO_ANALYZE = 5  # đổi thành 1 hoặc 10 nếu muốn xem ở cấu hình khác


def evaluate_confusion(predict_fn, class_to_images, novel_classes, k_shot, n_episodes=N_EPISODES):
    """Chạy nhiều episode, map ngược nhãn tương đối (0..N-1 trong episode)
    về đúng tên lớp thật, gom lại thành 2 danh sách y_true/y_pred để đưa
    vào sklearn.confusion_matrix."""
    import random
    random.seed(SEED)

    y_true, y_pred = [], []
    for _ in range(n_episodes):
        support_x, support_y, query_x, query_y, episode_classes = sample_episode(
            class_to_images, novel_classes, n_way=N_WAY, k_shot=k_shot,
            transform_support=eval_transform, transform_query=eval_transform,
        )
        pred_idx = predict_fn(support_x, support_y, query_x)

        for true_i, pred_i in zip(query_y.cpu().tolist(), pred_idx.cpu().tolist()):
            y_true.append(episode_classes[true_i])
            y_pred.append(episode_classes[pred_i])

    return y_true, y_pred


def plot_confusion(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # tỉ lệ theo hàng (recall từng lớp)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Nhãn dự đoán')
    ax.set_ylabel('Nhãn thật')
    ax.set_title(title, fontsize=11)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                    ha='center', va='center', fontsize=7.5, color=text_color)

    fig.colorbar(im, label='Tỉ lệ theo hàng')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {out_path}")

    print("\n" + classification_report(y_true, y_pred, labels=labels, zero_division=0))


def main():
    set_seed(SEED)
    meta = load_meta()
    class_to_images = meta['class_to_images']
    novel_classes = meta['novel_classes']

    # ---- ProtoNet ----
    print("=" * 60)
    print(f"PROTOTYPICAL NETWORK — {K_SHOT_TO_ANALYZE}-shot")
    print("=" * 60)
    model = Conv4Backbone().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, 'protonet_conv4_best_rngfix_20261014.pt'), map_location=DEVICE))
    model.eval()

    def protonet_predict(support_x, support_y, query_x):
        with torch.no_grad():
            log_p_y = protonet_forward(model, support_x, support_y, query_x, N_WAY)
            return log_p_y.argmax(dim=1)

    y_true_pn, y_pred_pn = evaluate_confusion(
        protonet_predict, class_to_images, novel_classes, K_SHOT_TO_ANALYZE)
    plot_confusion(y_true_pn, y_pred_pn, novel_classes,
                    f"Ma trận nhầm lẫn — Prototypical Network ({K_SHOT_TO_ANALYZE}-shot)",
                    os.path.join(RESULTS_DIR, f'confusion_protonet_{K_SHOT_TO_ANALYZE}shot_rngfix_20261014.png'))

    # ---- Fine-tuning (dùng đúng cấu hình đã tune) ----
    print("=" * 60)
    print(f"FINE-TUNING — {K_SHOT_TO_ANALYZE}-shot")
    print("=" * 60)
    # Dùng cấu hình đã chọn cho benchmark trong báo cáo
    ft_steps, ft_lr = 20, 1e-3
    print(f"Dùng cấu hình báo cáo: ft_steps={ft_steps}, ft_lr={ft_lr}")

    base_model = build_pretrained_backbone()

    def finetune_predict(support_x, support_y, query_x):
        classifier = finetune_on_episode(base_model, support_x, support_y, N_WAY,
                                          ft_steps=ft_steps, ft_lr=ft_lr)
        with torch.no_grad():
            query_feat = base_model(query_x)
            logits = classifier(query_feat)
            return logits.argmax(dim=1)

    y_true_ft, y_pred_ft = evaluate_confusion(
        finetune_predict, class_to_images, novel_classes, K_SHOT_TO_ANALYZE)
    plot_confusion(y_true_ft, y_pred_ft, novel_classes,
                    f"Ma trận nhầm lẫn — Fine-tuning ({K_SHOT_TO_ANALYZE}-shot)",
                    os.path.join(RESULTS_DIR, f'confusion_finetune_{K_SHOT_TO_ANALYZE}shot_rngfix_20261014.png'))


if __name__ == '__main__':
    main()
