# -*- coding: utf-8 -*-
"""
train_protonet.py
Giai đoạn 3 — Người 1: Prototypical Network

Yêu cầu chạy trước file này (đúng thứ tự):
1. data_prep.py  — đã tạo xong class_to_images.json trong Drive
2. episode_utils.py — nằm cùng thư mục, import được

Cách chạy trong Colab: copy nội dung file này vào 1 cell (hoặc nhiều cell
theo từng phần đánh số 1-4), chạy train() rồi evaluate(model).
"""

import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_prep import set_seed, N_WAY, K_SHOT, Q_QUERY, SEED
from episode_utils import (
    DEVICE, MODELS_DIR, RESULTS_DIR, load_meta, sample_episode,
    evaluate_episodes, save_result, train_transform, eval_transform,
)

N_TRAIN_EPISODES_PER_EPOCH = 100
N_EPOCHS = 60
LR = 1e-3
EVAL_EPISODES = 600  # chuẩn phổ biến trong paper few-shot learning

# --- Cấu hình cho phần validation (còn thiếu ở bản trước) ---
VAL_EVERY_N_EPOCHS = 5    # cứ mỗi 5 epoch thì đánh giá 1 lần trên validation classes
N_VAL_EPISODES = 100      # ít hơn 600 vì chỉ cần theo dõi xu hướng, không phải kết quả cuối cùng


# ============================================================
# 1. BACKBONE — Conv-4, kiến trúc chuẩn trong few-shot learning
# ============================================================

def conv_block(in_channels, out_channels):
    """1 khối: Conv2D 3x3 -> BatchNorm -> ReLU -> MaxPool2x2"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class Conv4Backbone(nn.Module):
    """Backbone chuẩn Conv-4: 4 khối Conv, mỗi khối 64 filter.
    Dùng AdaptiveAvgPool để luôn ra vector embedding cố định 64 chiều,
    không phụ thuộc chính xác kích thước ảnh đầu vào."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, embedding_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)           # (B, 64, H', W')
        x = self.pool(x)              # (B, 64, 1, 1)
        return x.view(x.size(0), -1)  # (B, 64)


# ============================================================
# 2. PROTOTYPICAL NETWORK — tính prototype + khoảng cách Euclidean
# ============================================================

def compute_prototypes(support_embeddings, support_labels, n_way):
    """Prototype mỗi lớp = trung bình embedding các ảnh support của lớp đó."""
    prototypes = []
    for c in range(n_way):
        mask = support_labels == c
        prototypes.append(support_embeddings[mask].mean(dim=0))
    return torch.stack(prototypes)  # (n_way, embedding_dim)


def euclidean_distance(a, b):
    """Khoảng cách Euclidean giữa mỗi hàng của a và mỗi hàng của b.
    a: (N, D), b: (M, D) -> trả về (N, M)"""
    n, m = a.size(0), b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a - b, 2).sum(dim=2)


def protonet_forward(model, support_x, support_y, query_x, n_way):
    """1 lần forward đầy đủ: encode ảnh -> tính prototype mỗi lớp -> tính
    khoảng cách -> trả về log-softmax của khoảng cách âm (dùng cho NLL loss)."""
    support_emb = model(support_x)
    query_emb = model(query_x)

    prototypes = compute_prototypes(support_emb, support_y, n_way)
    dists = euclidean_distance(query_emb, prototypes)  # (n_query, n_way)

    log_p_y = F.log_softmax(-dists, dim=1)
    return log_p_y


# ============================================================
# 3. VÒNG LẶP HUẤN LUYỆN (episodic training) — chỉ dùng base_classes
#    + đánh giá định kỳ trên validation_classes để chọn checkpoint tốt nhất
# ============================================================

def quick_validate(model, class_to_images, validation_classes, n_episodes=N_VAL_EPISODES):
    """Đánh giá nhanh trên validation classes (Pepper) — KHÔNG dùng để
    train, chỉ để theo dõi mô hình có đang tổng quát hoá tốt không.

    Lưu ý: Pepper chỉ có 2 lớp (Bacterial_spot, healthy), nên không thể
    chạy 5-way như lúc đánh giá cuối trên Durian (6 lớp). Ở đây dùng
    2-way — vẫn đủ để theo dõi xu hướng qua các epoch, dù không cùng
    cấu hình với kết quả benchmark cuối cùng."""
    val_n_way = min(N_WAY, len(validation_classes))
    if val_n_way < 2:
        print("  -> Bỏ qua validation: không đủ lớp.")
        return None

    predict_fn = make_predict_fn(model, val_n_way)
    result = evaluate_episodes(
        predict_fn, class_to_images, validation_classes,
        n_episodes=n_episodes, n_way=val_n_way, k_shot=K_SHOT, q_query=Q_QUERY,
    )
    return result['mean_accuracy']


def train():
    set_seed(SEED)
    meta = load_meta()
    class_to_images = meta['class_to_images']
    base_classes = meta['base_classes']
    validation_classes = meta['validation_classes']

    model = Conv4Backbone().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"Bắt đầu huấn luyện trên {DEVICE}, {len(base_classes)} base classes, "
          f"validate trên {len(validation_classes)} validation classes.")

    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')
    final_ckpt_path = os.path.join(MODELS_DIR, 'protonet_conv4_final.pt')
    history = []

    train_start_time = time.time()

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        for _ in range(N_TRAIN_EPISODES_PER_EPOCH):
            support_x, support_y, query_x, query_y, _ = sample_episode(
                class_to_images, base_classes,
                transform_support=train_transform, transform_query=train_transform,
            )

            log_p_y = protonet_forward(model, support_x, support_y, query_x, N_WAY)
            loss = F.nll_loss(log_p_y, query_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = log_p_y.argmax(dim=1)
                acc = (pred == query_y).float().mean().item()

            epoch_loss += loss.item()
            epoch_acc += acc

        scheduler.step()
        epoch_loss /= N_TRAIN_EPISODES_PER_EPOCH
        epoch_acc /= N_TRAIN_EPISODES_PER_EPOCH
        print(f"Epoch {epoch + 1}/{N_EPOCHS} — loss: {epoch_loss:.4f} — "
              f"acc (base, train): {epoch_acc:.4f}")

        # --- Đánh giá định kỳ trên validation classes ---
        val_acc = None
        is_last_epoch = (epoch + 1) == N_EPOCHS
        if (epoch + 1) % VAL_EVERY_N_EPOCHS == 0 or is_last_epoch:
            model.eval()
            val_acc = quick_validate(model, class_to_images, validation_classes)
            print(f"  -> Validation ({N_WAY}-way {K_SHOT}-shot, "
                  f"{N_VAL_EPISODES} episode): acc = {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  -> Validation accuracy tốt nhất mới, đã lưu "
                      f"checkpoint tại: {best_ckpt_path}")

        history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_acc': val_acc,
        })

    train_time_sec = time.time() - train_start_time

    # Luôn lưu thêm model ở epoch cuối cùng (để so sánh với "best" nếu cần
    # phân tích thêm), nhưng model dùng để đánh giá cuối cùng trên Durian
    # sẽ là bản "best" theo validation, không phải epoch cuối.
    torch.save(model.state_dict(), final_ckpt_path)

    training_summary = {
        'n_epochs': N_EPOCHS,
        'train_time_sec': train_time_sec,
        'train_time_min': train_time_sec / 60,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'best_ckpt_path': best_ckpt_path,
        'final_ckpt_path': final_ckpt_path,
        'history': history,
    }
    summary_path = os.path.join(RESULTS_DIR, 'protonet_training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, ensure_ascii=False, indent=2)

    print(f"\nHuấn luyện xong sau {train_time_sec / 60:.1f} phút.")
    print(f"Validation accuracy tốt nhất: {best_val_acc:.4f} tại epoch {best_epoch}")
    print(f"Đã lưu tóm tắt huấn luyện tại: {summary_path}")

    # Trả về model TỐT NHẤT theo validation (không phải model ở epoch cuối),
    # đây là model sẽ dùng để đánh giá cuối cùng trên Durian (novel classes)
    best_model = Conv4Backbone().to(DEVICE)
    best_model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    return best_model


# ============================================================
# 4. ĐÁNH GIÁ TRÊN NOVEL CLASSES (Durian) — dùng hạ tầng chung
# ============================================================

def make_predict_fn(model, n_way):
    """Bọc protonet_forward thành hàm predict_fn đúng chữ ký mà
    evaluate_episodes() trong episode_utils.py yêu cầu."""
    def predict_fn(support_x, support_y, query_x):
        model.eval()
        with torch.no_grad():
            log_p_y = protonet_forward(model, support_x, support_y, query_x, n_way)
            return log_p_y.argmax(dim=1)
    return predict_fn


def evaluate(model):
    meta = load_meta()
    class_to_images = meta['class_to_images']
    novel_classes = meta['novel_classes']

    # Đọc lại thời gian train đã lưu ở bước train(), để đính kèm vào mỗi
    # kết quả benchmark — Giai đoạn 5 cần so sánh thời gian giữa 2 phương pháp
    train_time_sec = None
    summary_path = os.path.join(RESULTS_DIR, 'protonet_training_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            train_time_sec = json.load(f).get('train_time_sec')

    for k in [1, 5, 10]:
        print(f"\n--- Đánh giá {N_WAY}-way {k}-shot trên novel classes (Durian) ---")
        predict_fn = make_predict_fn(model, N_WAY)
        result = evaluate_episodes(
            predict_fn, class_to_images, novel_classes,
            n_episodes=EVAL_EPISODES, k_shot=k,
        )
        result['method'] = 'ProtoNet'
        result['train_time_sec'] = train_time_sec
        print(f"Accuracy: {result['mean_accuracy']:.4f} ± {result['ci95']:.4f} "
              f"| thời gian inference TB: {result['mean_inference_time_sec']*1000:.2f} ms")
        save_result(result, f'protonet_{k}shot.json')


if __name__ == '__main__':
    trained_model = train()
    evaluate(trained_model)
