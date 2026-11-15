# -*- coding: utf-8 -*-
"""
audit_duplicates.py
Audit ảnh trùng chính xác (SHA-256) trên 6 lớp Durian.

Quét toàn bộ ảnh Durian (train / val / test), gom theo mã SHA-256, và ghi mọi
cặp ảnh giống hệt nhau ra CSV: cặp nằm ở hai tập khác nhau hay cùng một tập,
cùng nhãn hay khác nhãn. KHÔNG sửa, KHÔNG xoá ảnh nào, KHÔNG đổi nhãn nào.

Chỉ phát hiện bản sao giống hệt về byte; ảnh gần giống (cùng một lá chụp ở góc
khác, nén lại...) và nhãn sai chưa được nhận diện thì không nằm trong phạm vi.

Chạy trong Colab, sau data_prep.py (cần class_to_images.json):
    python audit_duplicates.py
hoặc:
    import audit_duplicates as ad
    ad.main()

Mặc định ghi vào results/exact_duplicate_audit_rerun.csv, KHÔNG ghi đè
exact_duplicate_audit.csv đã có. Muốn đổi tên file: ad.main(out_path=...).
Kết quả đúng với báo cáo phải ra: 11 cặp (khác tập hoặc khác nhãn), 9 cặp khác
tập, 2 cặp cùng tập, 5 cặp khác nhãn, 6 cặp cùng nhãn. Trên dữ liệu Durian script
còn ghi thêm 1 cặp cùng tập test cùng nhãn (Leaf_Algal_39 / Leaf_Algal_40) với
relevant=False: không gây rò rỉ nên không tính vào số liệu của báo cáo.
"""

import os
import csv
import hashlib
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from episode_utils import RESULTS_DIR, load_meta

AUDIT_CSV = os.path.join(RESULTS_DIR, 'exact_duplicate_audit_rerun.csv')

FIELDS = [
    'sha256', 'group_size', 'group_label_conflict', 'relation', 'same_label', 'relevant',
    'split_a', 'label_a', 'path_a', 'split_b', 'label_b', 'path_b',
]


def split_of(path):
    """Suy ra tập gốc từ đường dẫn .../<train|val|test>/<lớp>/<ảnh>."""
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3 and parts[-3] in ('train', 'val', 'test'):
        return parts[-3]
    return None


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def collect_images(meta):
    """[(đường dẫn, nhãn, tập)] cho toàn bộ ảnh của các lớp novel (Durian)."""
    items = []
    for c in meta['novel_classes']:
        for p in meta['class_to_images'][c]:
            items.append((p, c, split_of(p)))
    return items


def find_duplicate_groups(items, workers=16):
    """{sha256: [(tập, nhãn, đường dẫn), ...]} chỉ gồm các nhóm có từ 2 ảnh trở lên.
    Băm song song nhiều luồng vì đọc ảnh qua Google Drive chủ yếu là chờ I/O."""
    by_hash = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, h in enumerate(pool.map(lambda it: sha256_of(it[0]), items), 1):
            p, label, split = items[i - 1]
            by_hash[h].append((split, label, p))
            if i % 500 == 0 or i == len(items):
                print(f"  đã băm {i}/{len(items)} ảnh", flush=True)
    return {h: sorted(v, key=lambda t: t[2]) for h, v in by_hash.items() if len(v) > 1}


def build_rows(groups):
    """Một dòng cho mỗi cặp ảnh trong cùng nhóm hash."""
    rows = []
    for h in sorted(groups):
        members = groups[h]
        conflict = len({m[1] for m in members}) > 1
        for a, b in itertools.combinations(members, 2):
            rows.append({
                'sha256': h,
                'group_size': len(members),
                'group_label_conflict': conflict,
                'relation': 'cross_split' if a[0] != b[0] else 'same_split',
                'same_label': a[1] == b[1],
                # "relevant" = cặp có thể gây rò rỉ hoặc nhiễu nhãn: nằm ở hai tập khác nhau
                # hoặc có nhãn mâu thuẫn. Cặp cùng tập, cùng nhãn (vd. hai ảnh test giống
                # hệt cùng lớp) vẫn được ghi lại nhưng không tính vào số liệu của báo cáo.
                'relevant': (a[0] != b[0]) or conflict,
                'split_a': a[0], 'label_a': a[1], 'path_a': a[2],
                'split_b': b[0], 'label_b': b[1], 'path_b': b[2],
            })
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, n_images, n_groups):
    """Số liệu theo đúng phạm vi của báo cáo: chỉ tính các cặp relevant
    (khác tập hoặc khác nhãn). Số cặp trùng chính xác nói chung được in riêng."""
    rel = [r for r in rows if r['relevant']]
    cross = sum(r['relation'] == 'cross_split' for r in rel)
    same = sum(r['relation'] == 'same_split' for r in rel)
    diff_label = sum(not r['same_label'] for r in rel)
    print(f"Đã băm {n_images} ảnh. Số nhóm trùng chính xác: {n_groups} "
          f"({len(rows)} cặp, kể cả cặp cùng tập cùng nhãn).")
    print(f"Số cặp quan hệ theo phạm vi báo cáo (khác tập hoặc khác nhãn): {len(rel)}")
    print(f"  - khác tập (cross_split): {cross}")
    print(f"  - cùng tập (same_split):  {same}")
    print(f"  - khác nhãn: {diff_label}   cùng nhãn: {len(rel) - diff_label}")
    return {'pairs': len(rel), 'cross_split': cross, 'same_split': same,
            'different_label': diff_label, 'same_label': len(rel) - diff_label,
            'all_pairs': len(rows)}


def load_exclusions(csv_path=AUDIT_CSV):
    """Đọc file audit, trả về hai tập đường dẫn dùng cho giao thức đã lọc:
    - conflict: mọi ảnh thuộc nhóm hash có nhãn mâu thuẫn (loại khỏi cả train lẫn test);
    - test_dup: ảnh test có bản sao chính xác nằm trong train."""
    conflict, test_dup = set(), set()
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['group_label_conflict'] == 'True':
                conflict.add(r['path_a'])
                conflict.add(r['path_b'])
            pair = {r['split_a']: r['path_a'], r['split_b']: r['path_b']}
            if {r['split_a'], r['split_b']} == {'train', 'test'}:
                test_dup.add(pair['test'])
    return conflict, test_dup


def main(out_path=AUDIT_CSV, verbose=True):
    meta = load_meta()
    items = collect_images(meta)
    print(f"Đang tính SHA-256 cho {len(items)} ảnh...")
    groups = find_duplicate_groups(items)
    rows = build_rows(groups)

    if verbose:
        for h, members in groups.items():
            conflict = len({m[1] for m in members}) > 1
            multi = len({m[0] for m in members}) > 1
            print("=" * 60)
            print(f"SHA-256: {h}")
            print(f"Có trong nhiều tập: {multi}")
            print(f"Nhãn không nhất quán: {conflict}")
            for split, label, p in members:
                print(f"[{split}] [{label}] {p}")
        print("=" * 60)

    stats = summarize(rows, len(items), len(groups))
    write_csv(rows, out_path)
    print(f"\nĐã lưu: {out_path}")
    print("Không có ảnh nào bị sửa hay xoá.")
    return rows, stats


if __name__ == '__main__':
    main()
