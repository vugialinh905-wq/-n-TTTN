# -*- coding: utf-8 -*-
"""
episode_utils.py

So sánh 2 phương pháp công bằng, cả 2 PHẢI được đánh giá trên đúng
cùng 1 tập episode 
"""
import os
import json
import random
import time

import numpy as np
import torch

from data_prep import (
    IMG_SIZE, N_WAY, K_SHOT, Q_QUERY, SEED, CACHE_PATH, DRIVE_BASE_DIR,
    train_transform, eval_transform, load_and_preprocess, set_seed,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = os.path.normpath(os.path.join(DRIVE_BASE_DIR, '..', 'results'))
MODELS_DIR = os.path.normpath(os.path.join(DRIVE_BASE_DIR, '..', 'models'))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_meta():
    """ Đọc lại file cache đã tạo (data_prep.py)"""
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def effective_n_way(class_pool, desired_n_way=N_WAY, pool_name="tập lớp"):
    """Trả về n_way THỰC TẾ"""
    n = min(desired_n_way, len(class_pool))
    if n < desired_n_way:
        print(f"Lưu ý: {pool_name} chỉ có {len(class_pool)} lớp (< N_WAY="
              f"{desired_n_way} đã thống nhất ở Giai đoạn 0) — tạm dùng "
              f"{n}-way khi đánh giá trên {pool_name}. Không ảnh hưởng "
              f"đến kết quả đánh giá chính thức trên novel classes.")
    return n


def sample_episode(class_to_images, class_pool, n_way=N_WAY, k_shot=K_SHOT,
                    q_query=Q_QUERY, transform_support=None, transform_query=None):
    """Lấy ngẫu nhiên 1 episode: n_way lớp, mỗi lớp k_shot ảnh support +
    q_query ảnh query. Nhãn được đánh lại 0..n_way-1 TRONG PHẠM VI episode
    
    class_pool: danh sách tên lớp để rút mẫu (base_classes hoặc novel_classes)

    Trả về: support_x, support_y, query_x, query_y,
            và episode_classes (tên lớp gốc, để tra cứu khi cần debug)
    """
    if transform_support is None:
        transform_support = train_transform
    if transform_query is None:
        transform_query = eval_transform

    episode_classes = random.sample(class_pool, n_way)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for label_idx, cls in enumerate(episode_classes):# cls là tên lớp thật 
# label_idx là nhãn số trong episode, lớp đầu tiên -> 0, lớp thứ hai -> 1
        img_paths = class_to_images[cls]
        if len(img_paths) < k_shot + q_query:
            raise ValueError(
                f"Lớp '{cls}' chỉ có {len(img_paths)} ảnh, không đủ "
                f"K+Q={k_shot + q_query}. Kiểm tra lại report_class_counts() "
                f"ở data_prep.py trước khi chạy tiếp."
            )
        chosen = random.sample(img_paths, k_shot + q_query) # Chọn ngẫu nhiên K + Q ảnh
        support_paths = chosen[:k_shot] # K ảnh đầu làm support set
        query_paths = chosen[k_shot:]

        for p in support_paths:
            support_x.append(load_and_preprocess(p, transform_support))
            support_y.append(label_idx) # Gán nhãn số cho ảnh support
        for p in query_paths:
            query_x.append(load_and_preprocess(p, transform_query))
            query_y.append(label_idx) # Gán nhãn số cho ảnh query

    support_x = torch.stack(support_x).to(DEVICE) # Ghép list tensor thành 1 tensor lớn
    support_y = torch.tensor(support_y).to(DEVICE) # Chuyển danh sách nhãn thành tensor
    query_x = torch.stack(query_x).to(DEVICE)
    query_y = torch.tensor(query_y).to(DEVICE)

    return support_x, support_y, query_x, query_y, episode_classes


def evaluate_episodes(predict_fn, class_to_images, class_pool, n_episodes=600,
                       n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY, seed=SEED):
    """Hạ tầng đánh giá DÙNG CHUNG cho cả 2 phương pháp.

    predict_fn: hàm nhận (support_x, support_y, query_x) -> trả về nhãn dự
                đoán cho query_x (tensor int, cùng độ dài query_y). 

    Trả về: dict gồm accuracy trung bình, khoảng tin cậy 95%, thời gian
            inference trung bình mỗi episode (đơn vị giây).
    """
    random.seed(seed)  # cố định lại để 2 phương pháp thấy đúng CÙNG 1 chuỗi episode
    accuracies = []
    inference_times = []

    for ep in range(n_episodes):
        support_x, support_y, query_x, query_y, _ = sample_episode(
            class_to_images, class_pool, n_way, k_shot, q_query,
            transform_support=eval_transform,  # không augment lúc đánh giá
            transform_query=eval_transform,
        )

        t0 = time.time()
        pred_y = predict_fn(support_x, support_y, query_x) # Model dự đoán nhãn cho query set
        t1 = time.time()

        acc = (pred_y.cpu() == query_y.cpu()).float().mean().item() # Tính accuracy cho episode hiện tại
        accuracies.append(acc)
        inference_times.append(t1 - t0)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{n_episodes} — acc trung bình tạm thời: "
                  f"{np.mean(accuracies):.4f}")

    accuracies = np.array(accuracies) # Chuyển list accuracy thành mảng numpy
    mean_acc = accuracies.mean() # Accuracy trung bình của tất cả episode
    # Khoảng tin cậy 95% — cách báo cáo chuẩn trong các paper few-shot learning
    ci95 = 1.96 * accuracies.std() / np.sqrt(n_episodes)

    return {
        'n_episodes': n_episodes,
        'n_way': n_way,
        'k_shot': k_shot,
        'q_query': q_query,
        'mean_accuracy': float(mean_acc),
        'ci95': float(ci95),
        'mean_inference_time_sec': float(np.mean(inference_times)),
    }


def save_result(result, filename):
    """Lưu kết quả benchmark vào Drive (results/) để dùng chung vẽ biểu đồ
    so sánh."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả tại: {path}")
