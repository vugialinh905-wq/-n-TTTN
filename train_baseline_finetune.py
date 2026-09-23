# -*- coding: utf-8 -*-
"""
train_baseline_finetune.py

Baseline Fine-tuning dùng để so sánh với Prototypical Network.
Mỗi episode sẽ train nhanh một classifier trên support set,
sau đó đánh giá trên query set.
"""

# Thư viện hệ thống
import os
import json

# Thư viện Deep Learning
import torch
import torch.nn as nn
import torchvision.models as tvm

from data_prep import set_seed, N_WAY, K_SHOT, SEED

# Hàm hỗ trợ tạo và đánh giá episode
from episode_utils import (
    DEVICE,
    RESULTS_DIR,
    load_meta,
    evaluate_episodes,
    save_result,
    effective_n_way
)

# Số episode dùng để đánh giá cuối
EVAL_EPISODES = 600

# Giá trị mặc định (sẽ được tune lại trên validation)
FT_STEPS = 50
FT_LR = 1e-3

# Các giá trị thử nghiệm khi tune
FT_STEPS_CANDIDATES = [20, 50, 100]
FT_LR_CANDIDATES = [1e-3, 5e-3, 1e-2]

# Số episode dùng để tune
N_VAL_EPISODES = 100


# ============================================================
# 1. Tạo backbone MobileNetV2 pretrained
# ============================================================

def build_pretrained_backbone():
    """Tạo mạng trích xuất đặc trưng."""

    model = tvm.mobilenet_v2(
        weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1
    )  # tải trọng số pretrained từ ImageNet

    model.classifier = nn.Identity()  # bỏ classifier ImageNet

    model.eval()  # chế độ đánh giá

    for p in model.parameters():
        p.requires_grad = False  # khóa toàn bộ backbone

    return model.to(DEVICE)  # đưa model lên CPU/GPU


# ============================================================
# 2. Fine-tune classifier trên support set
# ============================================================

def finetune_on_episode(
    base_model,
    support_x,
    support_y,
    n_way,
    ft_steps=FT_STEPS,
    ft_lr=FT_LR
):
    """Train classifier mới bằng support set."""

    with torch.no_grad():
        support_feat = base_model(
            support_x
        )  # trích xuất đặc trưng từ support images

    feat_dim = support_feat.shape[1]  # số chiều vector đặc trưng

    classifier = nn.Linear(
        feat_dim,
        n_way
    ).to(DEVICE)  # classifier mới cho episode hiện tại

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=ft_lr
    )  # bộ tối ưu Adam

    criterion = nn.CrossEntropyLoss()  # hàm loss phân lớp

    classifier.train()  # bật chế độ train

    for _ in range(ft_steps):  # train nhiều vòng

        optimizer.zero_grad()  # xóa gradient cũ

        logits = classifier(
            support_feat
        )  # dự đoán nhãn support set

        loss = criterion(
            logits,
            support_y
        )  # tính sai số với nhãn thật

        loss.backward()  # tính gradient

        optimizer.step()  # cập nhật trọng số

    classifier.eval()  # chuyển sang chế độ đánh giá

    return classifier  # trả classifier đã học


# ============================================================
# 3. Tạo hàm predict cho evaluate_episodes()
# ============================================================

def make_predict_fn(
    base_model,
    n_way,
    ft_steps=FT_STEPS,
    ft_lr=FT_LR
):
    """Tạo hàm dự đoán cho từng episode."""

    def predict_fn(
        support_x,
        support_y,
        query_x
    ):

        classifier = finetune_on_episode(
            base_model,
            support_x,
            support_y,
            n_way,
            ft_steps=ft_steps,
            ft_lr=ft_lr
        )  # học từ support set

        with torch.no_grad():

            query_feat = base_model(
                query_x
            )  # trích xuất đặc trưng query images

            logits = classifier(
                query_feat
            )  # dự đoán nhãn query

            return logits.argmax(
                dim=1
            )  # lấy lớp có điểm cao nhất

    return predict_fn


# ============================================================
# 4. Tune FT_STEPS và FT_LR bằng validation set
# ============================================================

def tune_hyperparams_on_validation(
    base_model,
    class_to_images,
    validation_classes
):
    """Tìm bộ tham số cho accuracy cao nhất."""

    val_n_way = effective_n_way(
        validation_classes,
        desired_n_way=N_WAY,
        pool_name="validation_classes"
    )  # số lớp thực tế dùng để validation

    print(
        f"Tune trên {val_n_way}-way, "
        f"{N_VAL_EPISODES} episode..."
    )

    best_acc = -1.0  # accuracy tốt nhất hiện tại

    best_params = (
        FT_STEPS,
        FT_LR
    )  # tham số tốt nhất hiện tại

    tuning_log = []  # lưu kết quả thử nghiệm

    for steps in FT_STEPS_CANDIDATES:

        for lr in FT_LR_CANDIDATES:

            predict_fn = make_predict_fn(
                base_model,
                val_n_way,
                ft_steps=steps,
                ft_lr=lr
            )

            result = evaluate_episodes(
                predict_fn,
                class_to_images,
                validation_classes,
                n_episodes=N_VAL_EPISODES,
                n_way=val_n_way,
                k_shot=K_SHOT,
            )  # đánh giá trên validation

            acc = result['mean_accuracy']  # accuracy trung bình

            print(
                f"steps={steps}, "
                f"lr={lr:.0e} "
                f"-> acc={acc:.4f}"
            )

            tuning_log.append({
                'ft_steps': steps,
                'ft_lr': lr,
                'val_accuracy': acc
            })

            if acc > best_acc:  # nếu tốt hơn kết quả trước
                best_acc = acc
                best_params = (steps, lr)

    best_steps, best_lr = best_params

    print(
        f"Chọn: steps={best_steps}, "
        f"lr={best_lr:.0e}, "
        f"acc={best_acc:.4f}"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        'baseline_tuning_summary.json'
    )

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'candidates': tuning_log,
                'best_ft_steps': best_steps,
                'best_ft_lr': best_lr,
                'best_val_accuracy': best_acc,
            },
            f,
            ensure_ascii=False,
            indent=2
        )  # lưu kết quả tune

    print(f"Lưu log tại: {summary_path}")

    return best_steps, best_lr


# ============================================================
# 5. Đánh giá trên novel classes
# ============================================================

def evaluate():
    """Đánh giá mô hình trên tập novel."""

    set_seed(SEED)  # cố định seed

    meta = load_meta()  # đọc metadata

    class_to_images = meta['class_to_images']  # ảnh theo từng lớp

    novel_classes = meta['novel_classes']  # lớp dùng để test

    validation_classes = meta['validation_classes']  # lớp dùng để tune

    base_model = build_pretrained_backbone()  # tạo backbone

    best_ft_steps, best_ft_lr = tune_hyperparams_on_validation(
        base_model,
        class_to_images,
        validation_classes
    )  # chọn hyperparameter tốt nhất

    for k in [1, 5, 10]:

        print(
            f"\n--- Đánh giá {N_WAY}-way "
            f"{k}-shot ---"
        )

        predict_fn = make_predict_fn(
            base_model,
            N_WAY,
            ft_steps=best_ft_steps,
            ft_lr=best_ft_lr
        )  # tạo model với tham số tốt nhất

        result = evaluate_episodes(
            predict_fn,
            class_to_images,
            novel_classes,
            n_episodes=EVAL_EPISODES,
            k_shot=k,
        )  # đánh giá trên novel classes

        result['method'] = 'Fine-tuning (MobileNetV2)'
        result['ft_steps'] = best_ft_steps
        result['ft_lr'] = best_ft_lr

        print(
            f"Accuracy: "
            f"{result['mean_accuracy']:.4f} ± "
            f"{result['ci95']:.4f} | "
            f"Time: "
            f"{result['mean_inference_time_sec']*1000:.2f} ms"
        )

        save_result(
            result,
            f'baseline_finetune_{k}shot.json'
        )  # lưu kết quả ra file


# Điểm bắt đầu chương trình
if __name__ == '__main__':
    evaluate()
