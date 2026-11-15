# -*- coding: utf-8 -*-
"""
feature_analysis.py
Phân tích vì sao few-shot thấp trên Durian mà không huấn luyện lại. Đưa toàn bộ
ảnh Durian qua từng backbone một lần (lưu cache), rồi đánh giá few-shot bằng
prototype trên đặc trưng đã lưu (vài giây mỗi cấu hình).

    import feature_analysis as fa
    fa.build_cache()                                      # trích đặc trưng, chạy 1 lần
    fa.few_shot_grid(expected={1: 30.55, 5: 39.68, 10: 43.74})
    errs = fa.collect_errors('conv4_84', k=5)
    fa.top_confusions(errs)
    fa.show_errors(errs, 'Leaf_Blight', 'Leaf_Colletotrichum')
    fa.leakage_check()
    fa.robust_grid(ckpt=NEW)                              # so 3 giao thức: gộp / train->test / tránh ảnh gần trùng
"""

import os
import json
import time
import random
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

from data_prep import SEED, N_WAY, Q_QUERY, eval_transform, load_and_preprocess
from episode_utils import DEVICE, MODELS_DIR, RESULTS_DIR, load_meta
from train_protonet import Conv4Backbone
from train_baseline_finetune import build_pretrained_backbone

CACHE_DIR = os.path.join(RESULTS_DIR, 'feature_cache')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# tên cấu hình -> (loại backbone, kích thước ảnh đầu vào)
SETTINGS = {
    'conv4_84': ('conv4', 84),
    'mobilenet_84': ('mobilenet', 84),
    'mobilenet_128': ('mobilenet', 128),
    'mobilenet_224': ('mobilenet', 224),
}


def _default_ckpt():
    return os.path.join(MODELS_DIR, 'protonet_conv4_best.pt')


def _tf(size):
    if size == 84:
        return eval_transform  # đúng transform của pipeline chính
    return T.Compose([T.Resize((size, size)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _get_pool():
    """Toàn bộ ảnh Durian theo thứ tự lớp; kiểm tra khớp với danh sách lúc tạo cache."""
    meta = load_meta()
    classes = meta['novel_classes']
    paths = [p for c in classes for p in meta['class_to_images'][c]]
    labels = np.array([i for i, c in enumerate(classes) for _ in meta['class_to_images'][c]])
    idx_file = os.path.join(CACHE_DIR, 'index.json')
    if os.path.exists(idx_file):
        with open(idx_file, encoding='utf-8') as f:
            saved = json.load(f)
        if saved['paths'] != paths:
            raise RuntimeError("Danh sách ảnh khác lúc tạo cache. Xoá thư mục feature_cache "
                               "rồi chạy lại build_cache().")
    return meta, classes, paths, labels


def _feature_path(name, ckpt):
    if SETTINGS[name][0] == 'conv4':
        # gắn thời điểm sửa checkpoint vào tên file để cache không bị dùng nhầm bản cũ
        return os.path.join(CACHE_DIR, f'{name}_{int(os.path.getmtime(ckpt))}.npy')
    return os.path.join(CACHE_DIR, f'{name}.npy')


@torch.no_grad()
def _extract(model, paths, transform, batch_size=64):
    model.eval()
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([load_and_preprocess(p, transform)
                             for p in paths[i:i + batch_size]]).to(DEVICE)
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats)


def build_cache(ckpt=None):
    """Trích đặc trưng của mọi ảnh Durian cho từng cấu hình, lưu vào results/feature_cache/."""
    ckpt = ckpt or _default_ckpt()
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta, classes, paths, labels = _get_pool()
    mobilenet = None
    for name, (kind, size) in SETTINGS.items():
        out = _feature_path(name, ckpt)
        if os.path.exists(out):
            print(f"[{name}] đã có cache, bỏ qua.")
            continue
        if kind == 'conv4':
            model = Conv4Backbone().to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        else:
            if mobilenet is None:
                mobilenet = build_pretrained_backbone()
            model = mobilenet
        t0 = time.time()
        feats = _extract(model, paths, _tf(size))
        np.save(out, feats.astype(np.float32))
        print(f"[{name}] {feats.shape} xong sau {time.time() - t0:.0f}s")
    with open(os.path.join(CACHE_DIR, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump({'paths': paths, 'classes': classes}, f, ensure_ascii=False)


def load_features(name, ckpt=None):
    ckpt = ckpt or _default_ckpt()
    path = _feature_path(name, ckpt)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có {path}. Chạy build_cache() trước.")
    return np.load(path).astype(np.float64)


# ============================================================
# Đánh giá few-shot trên đặc trưng đã lưu
# ============================================================

def iter_episodes(class_to_images, class_pool, k, q, n_episodes, n_way=N_WAY, seed=SEED):
    """Rút episode theo đúng thứ tự gọi random của episode_utils.sample_episode,
    nên trùng từng episode với lần đánh giá chính thức."""
    random.seed(seed)
    for _ in range(n_episodes):
        classes = random.sample(class_pool, n_way)
        sup, qry = [], []
        for c in classes:
            chosen = random.sample(class_to_images[c], k + q)
            sup.append(chosen[:k])
            qry.append(chosen[k:])
        yield classes, sup, qry


def evaluate_features(feats, index, class_to_images, class_pool, k, metric='euclid',
                      n_episodes=600, q=Q_QUERY, n_way=N_WAY, seed=SEED, collect=False, episodes=None):
    """Prototype = trung bình đặc trưng support, dự đoán = prototype gần nhất (như ProtoNet).
    metric='cosine': chuẩn hoá L2 từng vector trước khi tính prototype.
    episodes: bộ sinh episode khác (mặc định: iter_episodes, đúng giao thức chính thức)."""
    X = feats
    if metric == 'cosine':
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    q_true = np.repeat(np.arange(n_way), q)

    accs, errors = [], []
    if episodes is None:
        episodes = iter_episodes(class_to_images, class_pool, k, q, n_episodes, n_way, seed)
    for classes, sup, qry in episodes:
        protos = np.stack([X[[index[p] for p in s]].mean(axis=0) for s in sup])
        q_paths = [p for qs in qry for p in qs]
        Q = X[[index[p] for p in q_paths]]
        d = (Q ** 2).sum(1)[:, None] - 2 * Q @ protos.T + (protos ** 2).sum(1)[None, :]
        pred = d.argmin(axis=1)
        accs.append((pred == q_true).mean())
        if collect:
            for i in np.where(pred != q_true)[0]:
                d_true, d_pred = d[i, q_true[i]], d[i, pred[i]]
                errors.append({
                    'path': q_paths[i], 'true': classes[q_true[i]], 'pred': classes[pred[i]],
                    'rel_margin': float((d_true - d_pred) / (d_true + d_pred + 1e-12)),
                })
    accs = np.array(accs)
    return {'acc': float(accs.mean()), 'ci95': float(1.96 * accs.std() / np.sqrt(n_episodes)),
            'errors': errors}


def few_shot_grid(ckpt=None, ks=(1, 5, 10), metrics=('euclid', 'cosine'),
                  n_episodes=600, expected=None):
    """In bảng accuracy 5-way theo cấu hình x K. expected={K: acc%} là số đánh giá chính
    thức của đúng checkpoint này, dùng để kiểm tra conv4_84/euclid có khớp không."""
    meta, classes, paths, labels = _get_pool()
    index = {p: i for i, p in enumerate(paths)}
    cti = meta['class_to_images']

    print(f"{'cấu hình':15s} {'độ đo':7s} | " + " | ".join(f"{k:>2}-shot      " for k in ks))
    conv4_euclid = {}
    for name in SETTINGS:
        feats = load_features(name, ckpt)
        for metric in metrics:
            cells = []
            for k in ks:
                r = evaluate_features(feats, index, cti, classes, k, metric, n_episodes)
                cells.append(f"{r['acc'] * 100:5.1f} ± {r['ci95'] * 100:3.1f}")
                if name == 'conv4_84' and metric == 'euclid':
                    conv4_euclid[k] = r['acc'] * 100
            print(f"{name:15s} {metric:7s} | " + " | ".join(cells))

    if expected and conv4_euclid:
        print("\nKiểm tra conv4_84/euclid so với số đánh giá chính thức (phải lệch dưới khoảng 0.3):")
        for k in ks:
            if k in expected and k in conv4_euclid:
                print(f"  {k:>2}-shot: cache {conv4_euclid[k]:.2f}  |  chính thức {expected[k]:.2f}"
                      f"  |  lệch {conv4_euclid[k] - expected[k]:+.2f}")


# ============================================================
# Xem ảnh bị đoán sai
# ============================================================

def collect_errors(name, k=5, metric='euclid', ckpt=None, n_episodes=600):
    meta, classes, paths, labels = _get_pool()
    index = {p: i for i, p in enumerate(paths)}
    r = evaluate_features(load_features(name, ckpt), index, meta['class_to_images'], classes,
                          k, metric, n_episodes, collect=True)
    print(f"{name}/{metric} {k}-shot: accuracy {r['acc'] * 100:.1f}%, {len(r['errors'])} lượt đoán sai")
    return r['errors']


def top_confusions(errors, n=8):
    c = Counter((e['true'], e['pred']) for e in errors)
    for (t, p), v in c.most_common(n):
        print(f"{t.replace('Leaf_', ''):15s} -> {p.replace('Leaf_', ''):15s} "
              f"{v:5d} lần ({v / len(errors) * 100:4.1f}% số lỗi)")


def show_errors(errors, true_class, pred_class, n=12, cols=4, out_path=None):
    """Lưới ảnh thật là true_class nhưng bị đoán thành pred_class, xếp theo mức "sai chắc chắn" giảm dần."""
    sel = [e for e in errors if e['true'] == true_class and e['pred'] == pred_class]
    seen, uniq = set(), []
    for e in sorted(sel, key=lambda e: -e['rel_margin']):
        if e['path'] not in seen:
            seen.add(e['path'])
            uniq.append(e)
    n_unique = len(uniq)
    uniq = uniq[:n]
    if not uniq:
        print("Không có ảnh nào.")
        return
    rows = (len(uniq) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows), dpi=110)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for ax, e in zip(axes, uniq):
        ax.imshow(Image.open(e['path']).convert('RGB'))
        ax.set_title(os.path.basename(e['path'])[:24], fontsize=7)
    fig.suptitle(f"Thật: {true_class}  |  bị đoán thành: {pred_class}  ({n_unique} ảnh khác nhau)", fontsize=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Kiểm tra ảnh train/test gốc của Durian có gần trùng không
# ============================================================

def split_of(path):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-3] in ('train', 'val', 'test'):
        return parts[-3]
    return None


def nearest_train_similarity(X, tr, te):
    """X đã chuẩn hoá L2. Với mỗi ảnh test: độ tương đồng cosine lớn nhất tới ảnh train và chỉ số ảnh đó."""
    sim = X[te] @ X[tr].T
    return sim.max(axis=1), sim.argmax(axis=1)


def leakage_check(name='mobilenet_224', n_show=6, ckpt=None, out_path=None):
    """Nếu nhiều ảnh test có ảnh train giống gần như y hệt thì số accuracy supervised bị lạc quan."""
    meta, classes, paths, labels = _get_pool()
    X = load_features(name, ckpt)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    splits = np.array([split_of(p) for p in paths], dtype=object)
    tr, te = np.where(splits == 'train')[0], np.where(splits == 'test')[0]
    if len(tr) == 0 or len(te) == 0:
        print("Không đọc được tập train/test gốc từ đường dẫn.")
        return
    best, arg = nearest_train_similarity(X, tr, te)
    print(f"{len(te)} ảnh test, {len(tr)} ảnh train. Độ tương đồng cosine tới ảnh train giống nhất:")
    for pct in (50, 90, 99):
        print(f"  phân vị {pct:>2}: {np.percentile(best, pct):.3f}")
    for thr in (0.99, 0.97, 0.95):
        print(f"  tỉ lệ ảnh test có ảnh train với độ tương đồng >= {thr}: {(best >= thr).mean() * 100:.1f}%")
    print(f"  1-NN (ảnh train giống nhất) đoán đúng lớp: {(labels[tr][arg] == labels[te]).mean() * 100:.1f}%")

    order = np.argsort(-best)[:n_show]
    fig, axes = plt.subplots(n_show, 2, figsize=(5, 2.6 * n_show), dpi=110)
    for r, o in enumerate(order):
        axes[r, 0].imshow(Image.open(paths[te[o]]).convert('RGB'))
        axes[r, 0].set_title(f"test | độ tương đồng {best[o]:.3f}", fontsize=8)
        axes[r, 1].imshow(Image.open(paths[tr[arg[o]]]).convert('RGB'))
        axes[r, 1].set_title("ảnh train giống nhất", fontsize=8)
        axes[r, 0].axis('off')
        axes[r, 1].axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Giao thức kiểm tra rò rỉ: tách tập gốc, tránh ảnh gần trùng
# ============================================================

def split_pools(class_to_images, classes):
    """{lớp: {'train': [...], 'test': [...]}} theo tập gốc của bộ dữ liệu (đọc từ đường dẫn)."""
    pools = {}
    for c in classes:
        pools[c] = {'train': [], 'test': []}
        for p in class_to_images[c]:
            s = split_of(p)
            if s in pools[c]:
                pools[c][s].append(p)
    return pools


def iter_split_episodes(pools, class_pool, k, q, n_episodes, n_way=N_WAY, seed=SEED):
    """Support lấy từ tập train gốc, query lấy từ tập test gốc."""
    for c in class_pool:
        if len(pools[c]['train']) < k or len(pools[c]['test']) < q:
            raise ValueError(f"{c}: train={len(pools[c]['train'])}, test={len(pools[c]['test'])} "
                             f"không đủ cho k={k}, q={q}.")
    random.seed(seed)
    for _ in range(n_episodes):
        classes = random.sample(class_pool, n_way)
        sup = [random.sample(pools[c]['train'], k) for c in classes]
        qry = [random.sample(pools[c]['test'], q) for c in classes]
        yield classes, sup, qry


def near_duplicate_groups(X, paths, labels, thr):
    """Gom các ảnh CÙNG LỚP có độ tương đồng cosine >= thr vào một nhóm (union-find).
    X đã chuẩn hoá L2. Trả về {đường dẫn: id nhóm}."""
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sim = X[idx] @ X[idx].T
        ii, jj = np.where(np.triu(sim >= thr, k=1))
        for a, b in zip(ii, jj):
            ra, rb = find(idx[a]), find(idx[b])
            if ra != rb:
                parent[ra] = rb
    return {p: find(i) for i, p in enumerate(paths)}


def iter_grouped_episodes(class_to_images, group_of, class_pool, k, q, n_episodes,
                          n_way=N_WAY, seed=SEED):
    """Như iter_episodes, nhưng query chỉ lấy từ ảnh không cùng nhóm gần trùng với ảnh support nào."""
    random.seed(seed)
    for _ in range(n_episodes):
        classes = random.sample(class_pool, n_way)
        sup, qry = [], []
        for c in classes:
            pool = class_to_images[c]
            s = random.sample(pool, k)
            s_groups = {group_of[p] for p in s}
            cand = [p for p in pool if group_of[p] not in s_groups]
            if len(cand) < q:
                raise ValueError(f"{c}: chỉ còn {len(cand)} ảnh query hợp lệ (cần {q}). "
                                 f"Ngưỡng gom nhóm quá thấp, nhiều ảnh bị gom chung.")
            sup.append(s)
            qry.append(random.sample(cand, q))
        yield classes, sup, qry


def _normalized(name, ckpt=None):
    X = load_features(name, ckpt)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def show_pairs_at_threshold(thr, name='mobilenet_224', n=6, ckpt=None, out_path=None):
    """Hiện các cặp cùng lớp có độ tương đồng NGAY TRÊN ngưỡng. Nếu đó vẫn là cùng một lá/cùng một
    ảnh thì ngưỡng an toàn; nếu là hai lá khác nhau thì ngưỡng quá thấp."""
    meta, classes, paths, labels = _get_pool()
    X = _normalized(name, ckpt)
    pairs = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sim = X[idx] @ X[idx].T
        ii, jj = np.where(np.triu(sim >= thr, k=1))
        pairs += [(float(sim[a, b]), idx[a], idx[b]) for a, b in zip(ii, jj)]
    print(f"Ngưỡng {thr}: {len(pairs)} cặp cùng lớp có độ tương đồng >= ngưỡng.")
    if not pairs:
        return
    pairs.sort(key=lambda t: t[0])
    chosen = pairs[:n]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(5, 2.6 * len(chosen)), dpi=110)
    axes = np.array(axes).reshape(len(chosen), 2)
    for r, (s, a, b) in enumerate(chosen):
        axes[r, 0].imshow(Image.open(paths[a]).convert('RGB'))
        axes[r, 0].set_title(f"{classes[labels[a]]} | {s:.3f}", fontsize=8)
        axes[r, 1].imshow(Image.open(paths[b]).convert('RGB'))
        axes[r, 0].axis('off')
        axes[r, 1].axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


def robust_grid(thr=0.97, names=('conv4_84', 'mobilenet_84', 'mobilenet_224'), metric='euclid',
                ks=(1, 5, 10), ckpt=None, n_episodes=600, dedup_feature='mobilenet_224'):
    """So accuracy 5-way dưới 3 giao thức: gộp train+val+test (chính thức), support=train/query=test,
    và tránh ảnh gần trùng. Nếu accuracy tụt mạnh ở hai giao thức sau thì kết quả chính thức bị lạc quan."""
    meta, classes, paths, labels = _get_pool()
    index = {p: i for i, p in enumerate(paths)}
    cti = meta['class_to_images']
    pools = split_pools(cti, classes)
    print("Số ảnh train/test gốc mỗi lớp:",
          {c.replace('Leaf_', ''): (len(pools[c]['train']), len(pools[c]['test'])) for c in classes})

    group_of = near_duplicate_groups(_normalized(dedup_feature, ckpt), paths, labels, thr)
    sizes = Counter(group_of.values())
    n_multi = sum(v for v in sizes.values() if v > 1)
    print(f"Ngưỡng gần trùng {thr}: {n_multi}/{len(paths)} ảnh nằm trong nhóm có từ 2 ảnh trở lên "
          f"(nhóm lớn nhất: {max(sizes.values())} ảnh).\n")

    protocols = [
        ('gộp (chính thức)', lambda k: None),
        ('support=train, query=test', lambda k: iter_split_episodes(pools, classes, k, Q_QUERY, n_episodes)),
        (f'tránh ảnh gần trùng ({thr})',
         lambda k: iter_grouped_episodes(cti, group_of, classes, k, Q_QUERY, n_episodes)),
    ]
    print(f"{'cấu hình':14s} {'giao thức':28s} | " + " | ".join(f"{k:>2}-shot     " for k in ks))
    for name in names:
        feats = load_features(name, ckpt)
        for pname, make in protocols:
            cells = []
            for k in ks:
                r = evaluate_features(feats, index, cti, classes, k, metric, n_episodes, episodes=make(k))
                cells.append(f"{r['acc'] * 100:5.1f} ± {r['ci95'] * 100:3.1f}")
            print(f"{name:14s} {pname:28s} | " + " | ".join(cells))
