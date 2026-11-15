# -*- coding: utf-8 -*-
"""
evaluate_clean_split.py
Đánh giá few-shot 5-way theo giao thức ĐÃ LỌC cho cả hai phương pháp
(nguồn của Bảng 5.1, Bảng G.1 và các ma trận nhầm lẫn ở Phụ lục G).

Giao thức:
  - support chỉ lấy từ tập train gốc của Durian, query chỉ lấy từ tập test gốc;
  - loại khỏi cả hai pool mọi ảnh thuộc nhóm SHA-256 có NHÃN MÂU THUẪN;
  - loại khỏi pool query các ảnh test có bản sao chính xác nằm trong train;
  - với mỗi K, sinh MỘT danh sách 600 episode và dùng chung cho cả hai phương pháp
    (cùng ảnh support, cùng ảnh query trong từng episode);
  - ProtoNet: checkpoint RNG-fix; Fine-tuning: MobileNetV2 đóng băng + Linear mới
    mỗi episode, 20 bước, lr 1e-3 (cấu hình của báo cáo, KHÔNG phải 100 bước / 5e-3).

Về tốc độ: cả hai backbone đều đóng băng và ở chế độ eval, ảnh chỉ qua eval_transform
(không ngẫu nhiên), nên đặc trưng của mỗi ảnh được tính MỘT lần rồi dùng lại cho mọi
episode. Kết quả như chạy lại backbone mỗi episode, nhưng nhanh hơn nhiều (chạy được
trên CPU). Phần huấn luyện lớp Linear của Fine-tuning vẫn chạy lại từ đầu mỗi episode.

Không sửa hay xoá ảnh nào; không ghi đè kết quả cũ (mặc định thêm hậu tố _rerun).

Chạy trong Colab, sau data_prep.py và audit_duplicates.py:
    python evaluate_clean_split.py
hoặc:
    import evaluate_clean_split as ev
    ev.main()                              # K = 1, 5, 10; ma trận nhầm lẫn ở 5-shot
    ev.main(ks=(5,), n_episodes=100)       # chạy thử nhanh

Số ra có thể lệch nhẹ (dưới khoảng tin cậy) so với báo cáo vì lớp Linear của
Fine-tuning khởi tạo ngẫu nhiên mỗi episode.
"""

import os
import csv
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from data_prep import SEED, N_WAY, Q_QUERY, eval_transform, load_and_preprocess, set_seed
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta
from train_protonet import Conv4Backbone, compute_prototypes, euclidean_distance
from train_baseline_finetune import build_pretrained_backbone, finetune_on_episode
from confusion_analysis import plot_confusion
from audit_duplicates import AUDIT_CSV, split_of, load_exclusions

CKPT = os.path.join(MODELS_DIR, 'protonet_conv4_best_rngfix_20261014.pt')
FT_STEPS = 20       # cấu hình của báo cáo (baseline_tuning_summary.json ban đầu)
FT_LR = 1e-3
EVAL_EPISODES = 600
KS = (1, 5, 10)
CONFUSION_K = 5


# ============================================================
# Pool đã lọc
# ============================================================

def build_pools(meta, conflict, test_dup):
    """{lớp: {'train': [...], 'test': [...]}} sau khi lọc, kèm bảng số ảnh trước/sau."""
    pools, table = {}, {}
    for c in meta['novel_classes']:
        tr = [p for p in meta['class_to_images'][c] if split_of(p) == 'train']
        te = [p for p in meta['class_to_images'][c] if split_of(p) == 'test']
        tr_keep = [p for p in tr if p not in conflict]
        te_keep = [p for p in te if p not in conflict and p not in test_dup]
        pools[c] = {'train': tr_keep, 'test': te_keep}
        table[c] = {'train_goc': len(tr), 'train_giu': len(tr_keep),
                    'test_goc': len(te), 'test_giu': len(te_keep)}
    return pools, table


def print_pool_table(table):
    print(f"{'Lớp':22s} {'Train gốc':>10s} {'Train giữ':>10s} {'Test gốc':>9s} {'Test giữ':>9s}")
    for c, r in table.items():
        print(f"{c:22s} {r['train_goc']:>10d} {r['train_giu']:>10d} "
              f"{r['test_goc']:>9d} {r['test_giu']:>9d}")


def iter_episodes(pools, classes, k, q, n_episodes, n_way=N_WAY, seed=SEED):
    """Support từ train, query từ test. Cùng seed -> cùng chuỗi episode."""
    random.seed(seed)
    for _ in range(n_episodes):
        chosen = random.sample(classes, n_way)
        sup = [random.sample(pools[c]['train'], k) for c in chosen]
        qry = [random.sample(pools[c]['test'], q) for c in chosen]
        yield chosen, sup, qry


# ============================================================
# Đặc trưng: tính một lần cho mỗi ảnh
# ============================================================

def load_images(paths, workers=16):
    """Nạp trước toàn bộ ảnh bằng nhiều luồng (đọc từ Google Drive chủ yếu là chờ I/O)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = pool.map(lambda q: load_and_preprocess(q, eval_transform), paths)
        tensors = {}
        for i, (p, t) in enumerate(zip(paths, loaded), 1):
            tensors[p] = t
            if i % 500 == 0 or i == len(paths):
                print(f"  đã nạp {i}/{len(paths)} ảnh", flush=True)
    return tensors


@torch.no_grad()
def extract_features(paths, proto_model, base_model, batch=128):
    """{đường dẫn: embedding Conv-4 (64 chiều)} và {đường dẫn: đặc trưng MobileNetV2 (1280 chiều)}."""
    tensors = load_images(paths)
    fp, ff = {}, {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.stack([tensors[p] for p in chunk]).to(DEVICE)
        ep, ef = proto_model(x).cpu(), base_model(x).cpu()
        for p, a, b in zip(chunk, ep, ef):
            fp[p], ff[p] = a, b
    print(f"  đã trích đặc trưng {len(paths)} ảnh", flush=True)
    return fp, ff


def proto_predict(sup_emb, sy, qry_emb, n_way):
    """Nhãn dự đoán của ProtoNet: prototype gần nhất (cùng phép tính với protonet_forward)."""
    protos = compute_prototypes(sup_emb, sy, n_way)
    return (-euclidean_distance(qry_emb, protos)).argmax(dim=1)


# ============================================================
# Đánh giá
# ============================================================

def ci95(acc):
    return float(1.96 * np.std(acc) / np.sqrt(len(acc)))


def evaluate_k(k, pools, classes, fp, ff, n_episodes, n_way=N_WAY, q=Q_QUERY):
    accs = {'protonet': [], 'finetune': []}
    y_true, y_pred = [], {'protonet': [], 'finetune': []}
    qy = torch.arange(n_way).repeat_interleave(q)
    sy = torch.arange(n_way).repeat_interleave(k).to(DEVICE)
    identity = nn.Identity()    # đặc trưng MobileNetV2 đã tính sẵn nên không qua backbone nữa

    def stack(feats, groups):
        return torch.stack([feats[p] for g in groups for p in g]).to(DEVICE)

    for ep, (chosen, sup, qry) in enumerate(
            iter_episodes(pools, classes, k, q, n_episodes, n_way), 1):
        with torch.no_grad():
            pred_p = proto_predict(stack(fp, sup), sy, stack(fp, qry), n_way).cpu()
        clf = finetune_on_episode(identity, stack(ff, sup), sy, n_way,
                                  ft_steps=FT_STEPS, ft_lr=FT_LR)
        with torch.no_grad():
            pred_f = clf(stack(ff, qry)).argmax(dim=1).cpu()

        for name, pred in (('protonet', pred_p), ('finetune', pred_f)):
            accs[name].append((pred == qy).float().mean().item())
            y_pred[name] += [chosen[i] for i in pred.tolist()]
        y_true += [chosen[i] for i in qy.tolist()]

        if ep % 100 == 0:
            print(f"  [K={k}] episode {ep}/{n_episodes} — ProtoNet "
                  f"{np.mean(accs['protonet']):.4f} | Fine-tuning {np.mean(accs['finetune']):.4f}")

    return accs, y_true, y_pred


def save_confusion(y_true, y_pred, labels, method, k, tag):
    title = {'protonet': 'Prototypical Network', 'finetune': 'Fine-tuning'}[method]
    base = f"confusion_{method}_clean_split_{k}shot_{tag}"
    png = os.path.join(RESULTS_DIR, base + '.png')
    plot_confusion(y_true, y_pred, labels,
                   f"Ma trận nhầm lẫn — {title} ({k}-shot, giao thức đã lọc)", png)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    csv_path = os.path.join(RESULTS_DIR, base + '.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['true\\pred'] + labels)
        for lab, row in zip(labels, cm):
            w.writerow([lab] + row.tolist())
    print(f"Đã lưu: {csv_path}")
    return f1_score(y_true, y_pred, labels=labels, average='macro'), \
        f1_score(y_true, y_pred, labels=labels, average=None).tolist()


def main(ckpt=CKPT, audit_csv=AUDIT_CSV, ks=KS, n_episodes=EVAL_EPISODES,
         tag='rerun', confusion_k=CONFUSION_K):
    set_seed(SEED)
    meta = load_meta()
    classes = meta['novel_classes']

    conflict, test_dup = load_exclusions(audit_csv)
    pools, table = build_pools(meta, conflict, test_dup)
    print(f"Ảnh thuộc nhóm nhãn mâu thuẫn: {len(conflict)}; "
          f"ảnh test trùng chính xác với train: {len(test_dup)}\n")
    print_pool_table(table)
    for c in classes:
        need_k, need_q = max(ks), Q_QUERY
        assert len(pools[c]['train']) >= need_k and len(pools[c]['test']) >= need_q, \
            f"{c}: không đủ ảnh sau khi lọc"

    proto = Conv4Backbone().to(DEVICE)
    proto.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    proto.eval()
    base = build_pretrained_backbone()
    print(f"\nProtoNet checkpoint: {ckpt}")
    print(f"Fine-tuning: MobileNetV2 đóng băng, {FT_STEPS} bước, lr {FT_LR}\n")
    fp, ff = extract_features([p for c in classes for p in pools[c]['train'] + pools[c]['test']],
                              proto, base)
    print()

    results = {'protonet': {}, 'finetune': {}}
    macro = {}
    for k in ks:
        print(f"--- {N_WAY}-way {k}-shot, {n_episodes} episode ---")
        accs, y_true, y_pred = evaluate_k(k, pools, classes, fp, ff, n_episodes)
        for name in ('protonet', 'finetune'):
            results[name][str(k)] = {
                'k_shot': k, 'n_episodes': n_episodes,
                'mean_accuracy': float(np.mean(accs[name])), 'ci95': ci95(accs[name]),
            }
            r = results[name][str(k)]
            print(f"{name:9s}: {r['mean_accuracy'] * 100:.2f}% ± {r['ci95'] * 100:.2f}%")
        if k == confusion_k:
            for name in ('protonet', 'finetune'):
                print(f"\n=== Ma trận nhầm lẫn: {name} ===")
                f1m, f1c = save_confusion(y_true, y_pred[name], classes, name, k, tag)
                macro[name] = {'macro_f1': float(f1m),
                               'f1_per_class': dict(zip(classes, map(float, f1c)))}
                print(f"Macro F1 = {f1m:.4f}")

    out = {
        'protocol': ('support từ train, query từ test; loại nhóm SHA-256 nhãn mâu thuẫn '
                     'và ảnh test trùng chính xác với train; cùng danh sách episode cho hai phương pháp'),
        'seed': SEED, 'n_way': N_WAY, 'q_query': Q_QUERY,
        'protonet_checkpoint': os.path.basename(ckpt),
        'finetune': {'ft_steps': FT_STEPS, 'ft_lr': FT_LR},
        'pool_sizes': table, 'results': results, f'confusion_{confusion_k}shot': macro,
    }
    path = os.path.join(RESULTS_DIR, f'fewshot_clean_split_audit_{tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu: {path}")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--audit-csv', default=AUDIT_CSV)
    ap.add_argument('--episodes', type=int, default=EVAL_EPISODES)
    ap.add_argument('--ks', type=int, nargs='+', default=list(KS))
    ap.add_argument('--tag', default='rerun')
    a = ap.parse_args()
    main(a.ckpt, a.audit_csv, tuple(a.ks), a.episodes, a.tag)
