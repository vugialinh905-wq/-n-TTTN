# -*- coding: utf-8 -*-
"""
train_baseline_finetune.py
Giai đoạn 3 — Người 2: Baseline Fine-tuning (đối chứng với Prototypical Network)

Nguyên tắc so sánh công bằng:
- Dùng ĐÚNG hạ tầng episode_utils.py (cùng cách rút episode, cùng seed,
  cùng số episode đánh giá) như bên Người 1 — xem evaluate_episodes().
- Với mỗi episode, model KHÔNG được biết trước các lớp đó — ta fine-tune
  nhanh CHỈ trên support set của episode đó (đúng K ảnh/lớp), rồi test
  trên query set của CHÍNH episode đó. Đây là cách so sánh công bằng nhất
  với few-shot learning: cùng điều kiện chỉ có K ảnh mỗi lớp để "học".

Yêu cầu chạy trước file này (đúng thứ tự):
1. data_prep.py     — đã tạo xong class_to_images.json trong Drive
2. episode_utils.py — nằm cùng thư mục, import được
"""

import os
import json

import torch
import torch.nn as nn
import torchvision.models as tvm

from data_prep import set_seed, N_WAY, K_SHOT, SEED
from episode_utils import DEVICE, RESULTS_DIR, load_meta, evaluate_episodes, save_result

EVAL_EPISODES = 600

# Giá trị mặc định — sẽ bị GHI ĐÈ bởi kết quả tune trên validation_classes
# (xem tune_hyperparams_on_validation), không dùng cố định như trước nữa.
FT_STEPS = 50   # số bước gradient fine-tune trên mỗi episode
FT_LR = 1e-3

# Lưới tham số thử trên validation — giữ nhỏ để không tốn quá nhiều thời
# gian, nhưng đủ để không chọn tham số "đoán mò"
FT_STEPS_CANDIDATES = [20, 50, 100]
FT_LR_CANDIDATES = [1e-3, 5e-3, 1e-2]
N_VAL_EPISODES = 100  # ít hơn 600 vì chỉ dùng để SO SÁNH các cấu hình, không phải kết quả cuối


# ============================================================
# 1. MODEL GỐC — MobileNetV2 pretrained trên ImageNet
# ============================================================

def build_pretrained_backbone():
    """Tải MobileNetV2 pretrained, bỏ lớp classifier gốc (1000 lớp
    ImageNet), giữ lại phần feature extractor để gắn classifier mới cho
    từng episode. Backbone được ĐÓNG BĂNG (không train lại toàn bộ mạng),
    vì K ảnh quá ít để fine-tune an toàn cả mạng lớn — chỉ train lớp
    classifier cuối, đây là cách làm hợp lý và phổ biến khi so sánh với
    few-shot learning trong điều kiện dữ liệu hạn chế."""
    model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


# ============================================================
# 2. FINE-TUNE NHANH TRÊN 1 EPISODE
# ============================================================

def finetune_on_episode(base_model, support_x, support_y, n_way,
                         ft_steps=FT_STEPS, ft_lr=FT_LR):
    """Gắn 1 lớp Linear mới (n_way classes) lên trên đặc trưng đã trích
    xuất, chỉ fine-tune lớp Linear này bằng đúng K ảnh/lớp của episode
    hiện tại. ft_steps/ft_lr có thể truyền riêng để dùng trong lúc tune
    hyperparameter trên validation, khác với giá trị mặc định cuối cùng
    dùng để đánh giá trên novel classes."""
    with torch.no_grad():
        support_feat = base_model(support_x)  # backbone đóng băng, trích 1 lần
    feat_dim = support_feat.shape[1]

    classifier = nn.Linear(feat_dim, n_way).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=ft_lr)
    criterion = nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(ft_steps):
        optimizer.zero_grad()
        logits = classifier(support_feat)
        loss = criterion(logits, support_y)
        loss.backward()
        optimizer.step()

    classifier.eval()
    return classifier


# ============================================================
# 3. HÀM PREDICT — đúng chữ ký evaluate_episodes() yêu cầu
#    (bao gồm CẢ thời gian fine-tune trong lúc đo inference time,
#    vì đây là chi phí thực tế phải trả khi gặp lớp mới — so sánh công
#    bằng với ProtoNet, vốn không cần fine-tune lại lúc gặp lớp mới)
# ============================================================

def make_predict_fn(base_model, n_way, ft_steps=FT_STEPS, ft_lr=FT_LR):
    def predict_fn(support_x, support_y, query_x):
        classifier = finetune_on_episode(base_model, support_x, support_y, n_way,
                                          ft_steps=ft_steps, ft_lr=ft_lr)
        with torch.no_grad():
            query_feat = base_model(query_x)
            logits = classifier(query_feat)
            return logits.argmax(dim=1)
    return predict_fn


# ============================================================
# 3b. TUNE FT_STEPS / FT_LR TRÊN VALIDATION — còn thiếu ở bản trước.
#     Trước đây 2 tham số này bị đặt cố định (đoán mò), trong khi bên
#     ProtoNet của Người 1 đã dùng validation_classes để chọn checkpoint
#     tốt nhất. Nếu baseline không được "tune" tương xứng, phần so sánh
#     ở Giai đoạn 5 sẽ THIÊN VỊ cho ProtoNet một cách không công bằng.
# ============================================================

def tune_hyperparams_on_validation(base_model, class_to_images, validation_classes):
    """Thử vài cấu hình (ft_steps, ft_lr) trên validation_classes (Pepper),
    chọn cấu hình có accuracy trung bình cao nhất — đúng vai trò của tập
    validation theo checklist Giai đoạn 0. Dùng K_SHOT mặc định (đã thống
    nhất ở Giai đoạn 0) làm điều kiện đại diện khi tune."""
    print(f"Đang tune FT_STEPS / FT_LR trên validation classes "
          f"({len(validation_classes)} lớp, {N_VAL_EPISODES} episode/cấu hình)...")

    best_acc = -1.0
    best_params = (FT_STEPS, FT_LR)
    tuning_log = []

    for steps in FT_STEPS_CANDIDATES:
        for lr in FT_LR_CANDIDATES:
            predict_fn = make_predict_fn(base_model, N_WAY, ft_steps=steps, ft_lr=lr)
            result = evaluate_episodes(
                predict_fn, class_to_images, validation_classes,
                n_episodes=N_VAL_EPISODES, k_shot=K_SHOT,
            )
            acc = result['mean_accuracy']
            print(f"  ft_steps={steps:3d}, ft_lr={lr:.0e} -> "
                  f"val acc = {acc:.4f}")
            tuning_log.append({'ft_steps': steps, 'ft_lr': lr, 'val_accuracy': acc})

            if acc > best_acc:
                best_acc = acc
                best_params = (steps, lr)

    best_steps, best_lr = best_params
    print(f"\nĐã chọn: FT_STEPS={best_steps}, FT_LR={best_lr:.0e} "
          f"(validation accuracy = {best_acc:.4f})")

    # Lưu lại toàn bộ quá trình tune để đưa vào báo cáo (đúng tinh thần
    # AI Development Log / giải trình phương pháp — không phải "đoán mò")
    summary_path = os.path.join(RESULTS_DIR, 'baseline_tuning_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'candidates': tuning_log,
            'best_ft_steps': best_steps,
            'best_ft_lr': best_lr,
            'best_val_accuracy': best_acc,
        }, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu tóm tắt quá trình tune tại: {summary_path}")

    return best_steps, best_lr


# ============================================================
# 4. CHẠY ĐÁNH GIÁ — dùng ĐÚNG hạ tầng chung (episode_utils.py)
# ============================================================

def evaluate():
    set_seed(SEED)
    meta = load_meta()
    class_to_images = meta['class_to_images']
    novel_classes = meta['novel_classes']
    validation_classes = meta['validation_classes']

    base_model = build_pretrained_backbone()

    # --- Bước còn thiếu ở bản trước: tune FT_STEPS/FT_LR trên validation ---
    best_ft_steps, best_ft_lr = tune_hyperparams_on_validation(
        base_model, class_to_images, validation_classes
    )

    for k in [1, 5, 10]:
        print(f"\n--- Đánh giá {N_WAY}-way {k}-shot (Fine-tuning baseline) ---")
        predict_fn = make_predict_fn(base_model, N_WAY,
                                      ft_steps=best_ft_steps, ft_lr=best_ft_lr)
        result = evaluate_episodes(
            predict_fn, class_to_images, novel_classes,
            n_episodes=EVAL_EPISODES, k_shot=k,
        )
        result['method'] = 'Fine-tuning (MobileNetV2)'
        result['ft_steps'] = best_ft_steps
        result['ft_lr'] = best_ft_lr
        print(f"Accuracy: {result['mean_accuracy']:.4f} ± {result['ci95']:.4f} "
              f"| thời gian TB mỗi episode (gồm cả fine-tune): "
              f"{result['mean_inference_time_sec']*1000:.2f} ms")
        save_result(result, f'baseline_finetune_{k}shot.json')


if __name__ == '__main__':
    evaluate()
