# -*- coding: utf-8 -*-
"""
domain_gap_diagnostics.py
Hai thí nghiệm đối chứng để tách nguyên nhân khiến few-shot trên Durian thấp:

1) supervised_ceiling(): huấn luyện bộ phân loại thường trên TOÀN BỘ ảnh train
   của Durian, đo trên tập test của Durian. Cho biết các lớp có tách được
   khi đủ nhãn hay không, và độ phân giải 84x84 có làm mất chi tiết không.
2) two_way_domain_gap(): đo ProtoNet ở chế độ 2-way trên Durian rồi so với
   Pepper (cùng N, cùng K, cùng mô hình, chỉ khác miền dữ liệu).
"""

import os
import io
import itertools
import contextlib

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data_prep import set_seed, SEED, eval_transform, load_and_preprocess, extract_datasets_to_local
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta, evaluate_episodes
from train_protonet import Conv4Backbone, make_predict_fn
from train_baseline_finetune import build_pretrained_backbone

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform_224 = T.Compose([
    T.Resize((224, 224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

PROTONET_CKPT = os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')


# ============================================================
# Chia train/val/test của Durian theo đúng tập gốc của bộ dữ liệu
# ============================================================

def split_of(path):
    """Suy ra tập gốc từ đường dẫn .../<train|val|test>/<lớp>/<ảnh>."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-3] in ('train', 'val', 'test'):
        return parts[-3]
    return None


def make_splits(class_to_images, classes):
    """Trả về {tên_tập: [(đường_dẫn, nhãn_số), ...]}. Nếu không đọc được tên
    tập gốc thì chia ngẫu nhiên 70/30 (có cảnh báo, vì có thể lọt ảnh gần trùng)."""
    by_split = {'train': [], 'val': [], 'test': []}
    items = []
    unknown = False
    for label, c in enumerate(classes):
        for p in class_to_images[c]:
            items.append((p, label))
            s = split_of(p)
            if s is None:
                unknown = True
            else:
                by_split[s].append((p, label))

    if unknown or not by_split['train'] or not by_split['test']:
        print("CẢNH BÁO: không đọc được tập train/test gốc từ đường dẫn, chia ngẫu nhiên "
              "70/30. Kết quả có thể lạc quan do ảnh gần trùng nằm ở cả hai tập.")
        paths = [p for p, _ in items]
        labels = [y for _, y in items]
        tr_p, te_p, tr_y, te_y = train_test_split(
            paths, labels, test_size=0.3, stratify=labels, random_state=SEED)
        return {'train': list(zip(tr_p, tr_y)), 'test': list(zip(te_p, te_y))}

    return {k: v for k, v in by_split.items() if v}


@torch.no_grad()
def extract_features(model, paths, transform, batch_size=64):
    """Đưa ảnh qua backbone, trả về ma trận đặc trưng (n_ảnh, chiều)."""
    model.eval()
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([
            load_and_preprocess(p, transform) for p in paths[i:i + batch_size]
        ]).to(DEVICE)
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats)


def plot_cm(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Nhãn dự đoán')
    ax.set_ylabel('Nhãn thật')
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.0f}%)",
                    ha='center', va='center', fontsize=7.5, color=color)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()


# ============================================================
# Thí nghiệm 1 — trần khi đủ nhãn
# ============================================================

def supervised_ceiling():
    meta = load_meta()
    classes = meta['novel_classes']
    splits = make_splits(meta['class_to_images'], classes)
    print("Số ảnh mỗi tập:", {k: len(v) for k, v in splits.items()})
    print(f"Ngẫu nhiên (6 lớp): {100 / len(classes):.1f}%\n")

    protonet = Conv4Backbone().to(DEVICE)
    protonet.load_state_dict(torch.load(PROTONET_CKPT, map_location=DEVICE))
    mobilenet = build_pretrained_backbone()

    settings = [
        ("MobileNetV2, ảnh 84x84", mobilenet, eval_transform, 'mobilenet84'),
        ("MobileNetV2, ảnh 224x224", mobilenet, transform_224, 'mobilenet224'),
        ("Conv-4 của ProtoNet, ảnh 84x84", protonet, eval_transform, 'conv4_84'),
    ]

    summary = []
    for name, model, tf, slug in settings:
        print("=" * 60)
        print(name)
        print("=" * 60)
        X = {s: extract_features(model, [p for p, _ in items], tf) for s, items in splits.items()}
        y = {s: np.array([l for _, l in items]) for s, items in splits.items()}

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(X['train'], y['train'])
        pred = clf.predict(X['test'])
        acc = accuracy_score(y['test'], pred)
        f1 = f1_score(y['test'], pred, average='macro')
        print(classification_report(y['test'], pred, target_names=classes, zero_division=0))

        out = os.path.join(RESULTS_DIR, f'supervised_cm_{slug}.png')
        plot_cm(y['test'], pred, classes, f"Supervised trên test Durian — {name}", out)
        print("Đã lưu:", out)
        summary.append((name, acc, f1))

    print("\n" + "=" * 60)
    print("TỔNG HỢP (bộ phân loại thường, đủ nhãn, đo trên test Durian)")
    print("=" * 60)
    for name, acc, f1 in summary:
        print(f"{name:34s} accuracy {acc * 100:5.1f}%   macro-F1 {f1 * 100:5.1f}%")


# ============================================================
# Thí nghiệm 2 — so sánh 2-way giữa Pepper và Durian
# ============================================================

def two_way_domain_gap(k_shot=5, eps_pepper=600, eps_pair=100):
    meta = load_meta()
    cti = meta['class_to_images']

    model = Conv4Backbone().to(DEVICE)
    model.load_state_dict(torch.load(PROTONET_CKPT, map_location=DEVICE))
    model.eval()
    print("Checkpoint:", PROTONET_CKPT)
    print("Sửa lần cuối:", __import__('time').ctime(os.path.getmtime(PROTONET_CKPT)))

    predict_fn = make_predict_fn(model, 2)
    seed = SEED + 123  # khác seed lúc chọn checkpoint, tránh dùng lại đúng các episode đó

    def run(pool, n_eps):
        with contextlib.redirect_stdout(io.StringIO()):
            r = evaluate_episodes(predict_fn, cti, pool, n_episodes=n_eps,
                                  n_way=2, k_shot=k_shot, seed=seed)
        return r['mean_accuracy'], r['ci95']

    pepper_acc, pepper_ci = run(meta['validation_classes'], eps_pepper)

    rows = []
    for a, b in itertools.combinations(meta['novel_classes'], 2):
        acc, ci = run([a, b], eps_pair)
        rows.append((a, b, acc, ci))
    rows.sort(key=lambda r: -r[2])

    print(f"\n{'Cặp lớp Durian (2-way, %d-shot)' % k_shot:58s} accuracy")
    for a, b, acc, ci in rows:
        print(f"{a + ' vs ' + b:58s} {acc * 100:5.1f}% ± {ci * 100:.1f}")

    healthy = [r for r in rows if 'Leaf_Healthy' in (r[0], r[1])]
    all_mean = np.mean([r[2] for r in rows])
    healthy_mean = np.mean([r[2] for r in healthy])

    print("\n" + "=" * 60)
    print(f"SO SÁNH 2-way {k_shot}-shot, cùng mô hình ProtoNet (ngẫu nhiên = 50%)")
    print("=" * 60)
    print(f"Pepper (cùng miền PlantVillage, healthy vs bệnh) : {pepper_acc * 100:5.1f}% ± {pepper_ci * 100:.1f}")
    print(f"Durian — 5 cặp Healthy vs từng bệnh              : {healthy_mean * 100:5.1f}%")
    print(f"Durian — cả 15 cặp                                : {all_mean * 100:5.1f}%")


def main():
    set_seed(SEED)
    extract_datasets_to_local()
    supervised_ceiling()
    two_way_domain_gap()


if __name__ == '__main__':
    main()
