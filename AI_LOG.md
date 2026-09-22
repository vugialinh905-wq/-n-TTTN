# AI Development Log

**Đề tài:** Nghiên cứu, đánh giá và xây dựng hệ thống nhận dạng bệnh trên lá
cây trồng bằng học ít mẫu và học chuyển giao (Few-shot Learning &
Transfer Learning) — Ứng dụng Prototypical Network cho các loài cây bản
địa Việt Nam
**Công cụ AI dùng xuyên suốt:** Claude (Anthropic), giao diện Claude.ai/app,
model Claude Sonnet 5, truy cập qua web/mobile chat interface

> ⚠️ Ghi chú cách dùng file này: các đoạn đánh dấu **⚠️ CẦN BẠN ĐIỀN** là chỗ
> cần điền ngày tháng/mã commit thật. Các đoạn **🔁 SINH VIÊN TỰ SỬA/VIẾT
> THÊM** là chỗ bạn/bạn cùng nhóm chỉnh sửa khác so với code AI đưa ban đầu —
> cần tự đối chiếu với code thật trên Git trước khi nộp. **📌 CẬP NHẬT MỚI**
> đánh dấu các phiên vừa bổ sung ở lần này.

---

## Phiên 1 — Định hướng và lựa chọn đề tài
*(nội dung giữ nguyên như bản trước — xem file gốc đã lưu trên Drive/Git)*

## Phiên 2 — Viết đề cương nghiên cứu
*(giữ nguyên như bản trước)*

## Phiên 3 — Thảo luận yêu cầu 3D của Frontend
*(giữ nguyên như bản trước)*

## Phiên 4 — Phân công công việc và kế hoạch chi tiết theo tuần
*(giữ nguyên như bản trước)*

## Phiên 5 — Chuẩn bị dữ liệu (`data_prep.py`, Giai đoạn 2)
*(giữ nguyên như bản trước — lưu ý: hàm `filter_corrupt_images()` nhắc
đến trong log gốc đã được xác nhận có tồn tại trong `data_prep.py` qua
thông báo lỗi thực tế ở Phiên 13 bên dưới, dù nguyên nhân lỗi lần đó
không phải do ảnh hỏng thật)*

## Phiên 6 — Xây hạ tầng Giai đoạn 3: episode_utils, ProtoNet, Baseline
*(giữ nguyên như bản trước)*

## Phiên 7 — Debug lỗi `quick_validate()` (N_WAY vs số lớp validation)
*(giữ nguyên như bản trước — xem thêm Phiên 11 bên dưới, đây là lần sửa
tạm thời đầu tiên, sau đó được thay thế bằng giải pháp tổng quát hơn)*

## Phiên 8 — Debug môi trường Colab/Drive/Kaggle (nhiều lỗi liên tiếp)
*(giữ nguyên như bản trước)*

## Phiên 9 — Thiết lập Git/GitHub và quy trình làm việc nhóm
*(giữ nguyên như bản trước)*

## Phiên 10 — Rà soát tuân thủ Quy định Thực tập tốt nghiệp
*(giữ nguyên như bản trước)*

---

## 📌 Phiên 11 — Tổng quát hóa cách xử lý N_WAY cho validation classes

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Thay vì sửa riêng lẻ lỗi N_WAY ở từng file (như Phiên 7 đã
  làm tạm cho `train_protonet.py`), xây 1 giải pháp dùng chung cho cả 2
  nhánh, vì Người 2 gặp lại đúng lỗi tương tự khi tự thêm bước tune
  hyperparameter trên validation classes.
- **Ngữ cảnh:** validation_classes (Pepper, PlantVillage) chỉ có 2 lớp,
  trong khi N_WAY=5 đã thống nhất từ Giai đoạn 0 — bất kỳ chỗ nào rút mẫu
  N_WAY từ validation_classes đều có nguy cơ lỗi `ValueError: Sample
  larger than population`.
- **Ngữ cảnh kỹ thuật:** Phiên này được thực hiện trên **giao diện
  claude.ai riêng** (không phải trong cuộc trò chuyện chính), sau đó ảnh
  chụp kết quả được đưa vào cuộc trò chuyện chính để rà soát lại.
- **Prompt gốc:** ⚠️ CẦN BẠN ĐIỀN (không có trong cuộc trò chuyện chính,
  chỉ thấy kết quả qua ảnh chụp màn hình)
- **File/thành phần liên quan:** `episode_utils.py` (thêm hàm mới),
  `train_protonet.py`, `train_baseline_finetune.py` (cả 2 đều sửa lại để
  dùng hàm mới)
- **Kết quả AI trả về:** Hàm `effective_n_way(class_pool, desired_n_way,
  pool_name)` — tự động tính `min(desired_n_way, len(class_pool))`, in
  cảnh báo rõ ràng thay vì để crash. `train_protonet.py` sửa
  `quick_validate()` và log trong `train()` để dùng hàm này (tính 1 lần,
  tránh lặp cảnh báo mỗi 5 epoch). `train_baseline_finetune.py` thêm hẳn
  hàm mới `tune_hyperparams_on_validation()` — thử lưới 3×3 cấu hình
  (`FT_STEPS_CANDIDATES` × `FT_LR_CANDIDATES`) trên validation, chọn cấu
  hình tốt nhất, lưu log tune vào `baseline_tuning_summary.json`.
- **Phần chấp nhận:** Toàn bộ giải pháp `effective_n_way()` dùng chung.
- **Phần chỉnh sửa:** 🔁 SINH VIÊN TỰ SỬA/VIẾT THÊM — quyết định mở rộng
  từ "chỉ sửa lỗi crash" thành "thêm hẳn bước tune hyperparameter có hệ
  thống" cho baseline — đây là cải tiến vượt quá yêu cầu tối thiểu, tự
  đề xuất bởi sinh viên (thể hiện qua comment trong code: *"Nếu baseline
  không được tune tương xứng, phần so sánh ở Giai đoạn 5 sẽ THIÊN VỊ cho
  ProtoNet một cách không công bằng"*).
- **Lý do chỉnh sửa:** Đảm bảo tính công bằng khoa học khi so sánh 2
  phương pháp — ProtoNet đã có bước chọn checkpoint tốt nhất qua
  validation, nên baseline cũng cần được tune tương xứng, không thể dùng
  tham số cố định "đoán mò".
- **Cách kiểm thử:** Xác nhận qua log chạy thực tế ở Phiên 17 (chạy lại
  `train()` và `evaluate()`, không còn lỗi `ValueError`).
- **Commit tương ứng:** ⚠️ CẦN BẠN ĐIỀN

---

## 📌 Phiên 12 — Hướng dẫn cập nhật (thay thế) code trên GitHub

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Đưa 3 file đã sửa ở Phiên 11 (đã copy thay thế trên Drive)
  lên GitHub, không biết quy trình "thay thế" file trên Git khác gì so
  với thêm file mới.
- **Prompt gốc:** "giờ thêm 3 file mới này thay thế ở drive rồi, nhưng
  giờ muốn thay thế ở github làm thế nào xóa file cũ thêm file mới chỉ
  từng bước"
- **File/thành phần liên quan:** `episode_utils.py`, `train_protonet.py`,
  `train_baseline_finetune.py`
- **Kết quả AI trả về:** Giải thích không cần xóa/thêm thủ công — Git tự
  nhận diện file cùng tên là "modified" khi nội dung thay đổi; hướng dẫn
  quy trình `git status` → `git add` → `git commit` → `git push`.
- **Phần chấp nhận:** Toàn bộ quy trình.
- **Cách kiểm thử:** Kiểm tra trực tiếp trên GitHub sau khi push, xác
  nhận nội dung file đã cập nhật và cột "Ngày sửa đổi" thay đổi.
- **Commit tương ứng:** ⚠️ CẦN BẠN ĐIỀN

---

## 📌 Phiên 13 — Debug Giai đoạn 4 (Người 1): chạy trên CPU và thiếu giải nén dữ liệu session mới

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Chạy lại `train()` đầy đủ 60 epoch trong notebook Colab
  mới (`GDD4.N1.ipynb`), gặp lỗi ngay từ epoch đầu.
- **Ngữ cảnh:** Notebook mới chưa được cấu hình GPU; đồng thời quên gọi
  `extract_datasets_to_local()` trước khi train, dẫn đến ảnh chưa tồn
  tại thật sự trên ổ đĩa `/content/` của session mới dù file cache JSON
  vẫn "biết" đường dẫn ảnh.
- **Prompt gốc:** Dán ảnh chụp lỗi kèm câu hỏi "bi lỗi rồi"
- **File/thành phần liên quan:** Không sửa code — lỗi cấu hình môi
  trường (Runtime type, thứ tự cell).
- **Kết quả AI trả về:** Chẩn đoán 2 nguyên nhân đồng thời: (1) dòng log
  "Bắt đầu huấn luyện trên cpu" → cần đổi Runtime type sang GPU T4; (2)
  `FileNotFoundError` khi đọc ảnh → cần thêm `extract_datasets_to_local()`
  vào đầu mỗi session mới trước khi gọi `build_all_class_to_images()`
  hoặc `train()`.
- **Phần chấp nhận:** Cả 2 hướng khắc phục.
- **Cách kiểm thử:** Xác nhận qua log chạy lại thành công (dòng "Bắt đầu
  huấn luyện trên cuda", train đủ 60 epoch không lỗi).
- **Commit tương ứng:** Không áp dụng (thao tác môi trường, không đổi code).

---

## 📌 Phiên 14 — Đánh giá kết quả Giai đoạn 4 (Người 1) và phân tích domain shift

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Diễn giải ý nghĩa bảng kết quả sau khi train xong 60
  epoch (validation 86.40%, Durian 1/5/10-shot lần lượt 31.38% / 42.82% /
  46.81%).
- **Prompt gốc:** "gđ 3 n1 tới đây ok chưa" → "vậy giai đoạn 3 4 của tôi
  xong rồi hả"
- **File/thành phần liên quan:** Không có code — phân tích kết quả JSON
  đã lưu (`protonet_1shot.json`, `_5shot.json`, `_10shot.json`,
  `protonet_training_summary.json`).
- **Kết quả AI trả về:** Xác nhận Giai đoạn 3+4 của Người 1 đã hoàn tất
  về mặt kỹ thuật; giải thích chênh lệch lớn giữa accuracy validation
  (cùng domain PlantVillage) và novel classes Durian (khác domain) là
  hiện tượng **domain shift** — đây là phát hiện khoa học có giá trị,
  không phải lỗi, khớp đúng với động lực ban đầu chọn đề tài (data cây
  Việt Nam khan hiếm và khác biệt domain).
- **Phần chấp nhận:** Cách diễn giải domain shift, dùng làm luận điểm
  chính trong báo cáo.
- **Cách kiểm thử:** Không áp dụng (phân tích số liệu đã có).
- **Commit tương ứng:** Không áp dụng.

---

## 📌 Phiên 15 — Đối chiếu tiến độ với tên đề tài chính thức đã chốt

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Xác nhận thiết kế 2 nhánh (ProtoNet / Fine-tuning) có
  khớp đúng với tên đề tài thầy đã chốt hay không, và xác định phần công
  việc còn thiếu.
- **Prompt gốc:** "người 2 hoàn thành giai đoạn 4 xong, giờ chúng tôi
  hoàn thành xong code chính chưa nhỉ..."
- **Kết quả AI trả về:** Xác nhận 2 nhánh ánh xạ đúng 1-1 vào tên đề tài
  ("học ít mẫu" = Few-shot/ProtoNet, "học chuyển giao" = Transfer
  Learning/Fine-tuning); chỉ ra rõ phần **"xây dựng hệ thống"** (demo ứng
  dụng) trong tên đề tài **chưa được thực hiện**, cần làm ở Giai đoạn 6,
  không nên báo cáo với thầy là "đã xong 100%".
- **Phần chấp nhận:** Toàn bộ nhận định — dùng làm cơ sở chuẩn bị nội
  dung trao đổi với giảng viên hướng dẫn.
- **Cách kiểm thử:** Không áp dụng.
- **Commit tương ứng:** Không áp dụng.

---

## 📌 Phiên 16 — Rà soát file kiến trúc mới `stage4_run_all.py`

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Đánh giá 1 file kiến trúc pipeline mới (tách train/evaluate
  thành 4 bước độc lập qua `argparse --stage`), sinh viên nhận được từ
  nguồn khác (không rõ từ phiên AI nào) và muốn biết có nên áp dụng
  không.
- **Ngữ cảnh:** ⚠️ CẦN BẠN ĐIỀN — làm rõ file `stage4_run_all.py` này từ
  đâu ra (phiên claude.ai khác, hay tự viết, hay nguồn khác) để log được
  chính xác.
- **Prompt gốc:** Dán toàn bộ nội dung file `stage4_run_all.py`, hỏi
  "coi gđ 4 người 2 làm ổn chưa"
- **File/thành phần liên quan:** `stage4_run_all.py` (file mới, chưa
  từng tồn tại trước đó)
- **Kết quả AI trả về:** Phát hiện file gọi đến 4 hàm **chưa hề tồn tại**
  trong các file gốc (`load_trained_model()`, `train_and_save()`,
  `load_trained_config()`, và toàn bộ module `compare_results.py`) — kết
  luận file này **chưa chạy được**, đồng thời có dấu hiệu viết cho môi
  trường khác (nhắc "README_VSCODE.md") không khớp với môi trường Colab
  nhóm đang dùng. Khuyến nghị **không áp dụng ngay**, giữ nguyên kết quả
  đã có từ kiến trúc gốc.
- **Phần chấp nhận:** Khuyến nghị không dùng file này, tiếp tục với
  `train_protonet.py` / `train_baseline_finetune.py` bản đã chạy thành
  công.
- **Phần loại bỏ:** Toàn bộ `stage4_run_all.py` — không đưa vào pipeline
  chính thức, có thể giữ lại như ý tưởng cho phần "hướng phát triển".
- **Lý do:** Tránh đổi kiến trúc giữa chừng khi đã có kết quả hợp lệ,
  tránh phát sinh thêm việc viết code mới không cần thiết ở giai đoạn gần
  cuối.
- **Cách kiểm thử:** Không áp dụng (không chạy thử file này).
- **Commit tương ứng:** Không áp dụng (file không được đưa vào repo
  chính thức) — ⚠️ CẦN BẠN XÁC NHẬN file này có bị lỡ commit vào Git chưa,
  nếu có cần cân nhắc xoá hoặc ghi chú rõ "chưa hoàn thiện, không dùng".

---

## 📌 Phiên 17 — Chạy lại Giai đoạn 3+4 Người 2 và debug 2 lỗi phát sinh

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Chạy lại toàn bộ `train_baseline_finetune.py` (bản đã
  sửa `effective_n_way` ở Phiên 11) trong notebook Colab mới, để có kết
  quả đáng tin cậy đầu tiên (lần chạy trước đó bị crash do lỗi N_WAY).
- **Prompt gốc:** "giờ tôi sẽ làm lại phần gđ 3 của người 2 bắt đầu thế
  nào chỉ từng bước"
- **File/thành phần liên quan:** `train_baseline_finetune.py`,
  `data_prep.py`, `episode_utils.py`
- **Kết quả AI trả về:** Hướng dẫn 6 bước (bật GPU, mount Drive + git
  pull, giải nén dữ liệu, chạy `evaluate()`, xác nhận kết quả, push Git).
- **Lỗi phát sinh & xử lý (2 lỗi liên tiếp trong cùng phiên):**
  1. `NameError: name 'load_meta' is not defined` — do hướng dẫn ban đầu
     của AI thiếu dòng `from episode_utils import *` (hàm `load_meta()`
     nằm ở `episode_utils.py`, không phải `data_prep.py`) — **lỗi này do
     AI hướng dẫn thiếu sót**, được sinh viên phát hiện và AI sửa ngay
     trong cùng phiên.
  2. `Author identity unknown` khi `git push` — do notebook mới chưa
     từng chạy `git config --global user.email/user.name` — AI hướng
     dẫn bổ sung 2 dòng config trước khi commit.
- **Kết quả chạy thành công:** Tune chọn được `FT_STEPS=20, FT_LR=1e-03`
  (validation accuracy 0.8183); đánh giá Durian: 1-shot 28.94% ± 0.47%,
  5-shot 46.47% ± 0.55%, 10-shot 53.61% ± 0.56%.
- **Cách kiểm thử:** Xác nhận qua log chạy thực tế trong Colab (ảnh chụp
  màn hình), đủ 8 file kết quả trong `RESULTS_DIR`.
- **Commit tương ứng:** ⚠️ CẦN BẠN ĐIỀN

---

## 📌 Phiên 18 — Tổng hợp, so sánh kết quả 2 phương pháp và trực quan hóa

- **Ngày:** ⚠️ CẦN BẠN ĐIỀN
- **Mục tiêu:** Gộp kết quả 2 nhánh thành 1 bảng so sánh, phân tích ý
  nghĩa, chuẩn bị nội dung cho Giai đoạn 5 và buổi trao đổi với giảng
  viên.
- **Prompt gốc:** "vậy có cần so sánh gì với phần của tôi không kết quả
  vậy ổn ch" → "ok xong"
- **File/thành phần liên quan:** Không sửa code — tổng hợp từ 8 file
  JSON kết quả đã có (4 file ProtoNet + 4 file Baseline).
- **Kết quả AI trả về:** Bảng so sánh Accuracy + thời gian theo từng
  cấu hình K-shot; biểu đồ đường (line chart) minh họa 2 đường accuracy
  cắt nhau; phân tích trade-off: ProtoNet vượt trội ở 1-shot và nhanh
  hơn ~2 lần, Fine-tuning vượt trội ở 5-shot/10-shot; đề xuất câu kết
  luận cho báo cáo (chiến lược "lai" tùy lượng dữ liệu có sẵn).
- **Phần chấp nhận:** Toàn bộ cách phân tích và luận điểm trade-off, dự
  kiến đưa vào phần Kết quả & Thảo luận của báo cáo.
- **Cách kiểm thử:** Không áp dụng (tổng hợp số liệu đã kiểm chứng ở các
  phiên trước).
- **Commit tương ứng:** Không áp dụng (chưa có code mới, chỉ là văn bản
  phân tích — ⚠️ nếu đưa bảng/biểu đồ này vào file báo cáo trong repo,
  cần ghi commit tương ứng).

---

## Việc cần làm trước khi nộp

1. Điền toàn bộ chỗ đánh dấu **⚠️ CẦN BẠN ĐIỀN** (ngày tháng thực tế, mã
   commit — dùng `git log --oneline` để lấy).
2. Đối chiếu các chỗ **🔁 SINH VIÊN TỰ SỬA/VIẾT THÊM** với code thật trên
   GitHub, sửa mô tả cho khớp chính xác.
3. **Xác nhận nguồn gốc file `stage4_run_all.py`** (Phiên 16) — làm rõ
   phiên AI nào tạo ra file này, và xác nhận nó không bị lẫn vào repo
   chính thức nếu không dùng.
4. Bổ sung chi tiết phần Người 2 tự viết (`tune_hyperparams_on_validation`,
   các hàm trong `train_baseline_finetune.py`) — nhờ Người 2 tự mô tả lại
   góc nhìn của họ cho đầy đủ, vì log này chủ yếu ghi từ phía Người 1.
5. Sau khi hoàn thiện, commit file này vào Git, ví dụ:
   `"Cap nhat AI Development Log den het Giai doan 4"`.
