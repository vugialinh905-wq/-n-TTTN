# -*- coding: utf-8 -*-
"""
confusion_matrix_protonet.py
Chẩn đoán domain shift: ma trận nhầm lẫn của ProtoNet trên TOÀN BỘ 6 lớp
Durian cùng lúc (6-way, khác với benchmark chính 5-way) để xem rõ lớp
nào bị nhầm với lớp nào.

Chạy trên Colab, sau khi đã có model đã train (protonet_conv4_best.pt).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data_prep import set_seed, K_SHOT, Q_QUERY, SEED
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta, sample_episode
from train_protonet import Conv4Backbone, protonet_forward

N_DIAGNOSTIC_EPISODES = 200  # ít hơn 600 vì đây là chẩn đoán, không phải benchmark chính


def load_best_model():
    model = Conv4Backbone().to(DEVICE)
    ckpt_path = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def collect_predictions(model, class_to_images, novel_classes, n_episodes=N_DIAGNOSTIC_EPISODES):
    """Chạy nhiều episode 6-way (dùng HẾT 6 lớp Durian mỗi lần, không phải
    5-way như benchmark chính), ghi lại nhãn thật và nhãn dự đoán theo
    TÊN LỚP THẬT (không phải chỉ số cục bộ trong episode)."""
    n_way = len(novel_classes)  # = 6, dùng hết toàn bộ lớp Durian
    y_true, y_pred = [], []

    for _ in range(n_episodes):
        support_x, support_y, query_x, query_y, episode_classes = sample_episode(
            class_to_images, novel_classes, n_way=n_way,
            k_shot=K_SHOT, q_query=Q_QUERY,
        )
        with torch.no_grad():
            log_p_y = protonet_forward(model, support_x, support_y, query_x, n_way)
            pred_local = log_p_y.argmax(dim=1).cpu().numpy()

        true_local = query_y.cpu().numpy()
        # Đổi chỉ số cục bộ (0..5) về tên lớp thật để gộp đúng qua nhiều episode
        y_true += [episode_classes[i] for i in true_local]
        y_pred += [episode_classes[i] for i in pred_local]

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # chuẩn hoá theo hàng = recall từng lớp

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Lớp dự đoán')
    ax.set_ylabel('Lớp thật')
    ax.set_title('Ma trận nhầm lẫn — ProtoNet trên 6 lớp Durian (chuẩn hoá theo hàng)')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i, j]*100:.0f}%\n({cm[i, j]})',
                     ha='center', va='center', color=color, fontsize=8)

    fig.colorbar(im, label='Tỷ lệ (theo hàng)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Đã lưu ma trận nhầm lẫn tại: {save_path}")
    return cm, cm_norm


def report_top_confusions(cm, cm_norm, class_names, top_n=5):
    """In ra top các cặp (lớp thật, lớp dự đoán nhầm) bị nhầm nhiều nhất,
    bỏ qua đường chéo (dự đoán đúng)."""
    pairs = []
    for i, true_c in enumerate(class_names):
        for j, pred_c in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                pairs.append((cm_norm[i, j], cm[i, j], true_c, pred_c))
    pairs.sort(reverse=True)

    print(f"\n--- Top {top_n} cặp nhầm lẫn nhiều nhất ---")
    for rate, count, true_c, pred_c in pairs[:top_n]:
        print(f"  Thật = {true_c:20s} -> Dự đoán nhầm = {pred_c:20s} "
              f"| {rate*100:.1f}% ({count} lần)")


def main():
    set_seed(SEED)
    model = load_best_model()
    meta = load_meta()
    class_to_images = meta['class_to_images']
    novel_classes = meta['novel_classes']

    print(f"Đang chạy {N_DIAGNOSTIC_EPISODES} episode 6-way trên Durian để thu thập dự đoán...")
    y_true, y_pred = collect_predictions(model, class_to_images, novel_classes)

    overall_acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
    print(f"Accuracy tổng thể (6-way, {K_SHOT}-shot, {N_DIAGNOSTIC_EPISODES} episode): {overall_acc*100:.2f}%")

    save_path = os.path.join(RESULTS_DIR, 'confusion_matrix_protonet_durian.png')
    cm, cm_norm = plot_confusion_matrix(y_true, y_pred, novel_classes, save_path)
    report_top_confusions(cm, cm_norm, novel_classes)


if __name__ == '__main__':
    main()
