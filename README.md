# Vietnamese Punctuation Restoration — Khôi phục dấu câu tiếng Việt

Khôi phục dấu câu cho văn bản tiếng Việt **không dấu câu, không viết hoa**,
bằng bài toán *word-level sequence labeling*.

```
input :  bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài
output:  Bạn đã hoàn thành bài tập chưa? Ngày mai chúng ta nộp bài.
```

> ## ✅ DỰ ÁN HOÀN CHỈNH
>
> Bốn thí nghiệm đã huấn luyện xong, model thắng cuộc đã được chọn **chỉ bằng
> validation**, official test đã chạy **đúng một lần**, inference và UI đã hoạt
> động.
>
> | | |
> |---|---|
> | **Winner** | **E2** — `vinai/phobert-base-v2`, không class weight |
> | **Validation Punctuation Macro-F1** | **0.7787** (chọn winner bằng con số này) |
> | **Official test Punctuation Macro-F1** | **0.7763** (đo một lần, sau khi khoá winner) |
> | Test accuracy | 0.9663 |
> | Test macro-F1 (4 lớp) | 0.8288 |
>
> Mọi con số trong README này đọc từ artifact thật trong `outputs/`. Không có
> kết quả nào được nhập tay.

---

## 1. Project Overview

Bài toán đầu-cuối: văn bản thô không dấu câu → mô hình gán một nhãn dấu câu cho
mỗi từ → dựng lại văn bản có dấu câu và viết hoa đầu câu.

Dự án được chia làm hai giai đoạn, và **thứ tự đó là điểm cốt lõi về phương
pháp luận**:

```
JointCapPunc (ghim commit)
        ↓
Data pipeline  →  train / validation / test
        ↓
E1 BiLSTM ─┐
E2 PhoBERT ─┤  bốn thí nghiệm, huấn luyện độc lập
E3 PhoBERT ─┤  (chỉ dùng train + validation)
E4 PhoBERT ─┘
        ↓
So sánh trên VALIDATION  →  chọn winner  →  KHOÁ winner
        ↓
Official test — chạy ĐÚNG MỘT LẦN trên winner
        ↓
Post-hoc comparison (mô tả, không đổi winner)
        ↓
Inference  →  Streamlit UI
```

Tập test không được nạp ở bất kỳ đâu cho đến khi winner đã bị khoá. Đây là điều
làm cho con số 0.7763 là một ước lượng **không thiên lệch**, chứ không phải con
số đẹp nhất tìm được sau nhiều lần thử.

## 2. Problem Definition

Mỗi *lexical word* nhận đúng một nhãn, mô tả dấu câu đứng **sau** từ đó:

| id | nhãn | ký hiệu | ý nghĩa |
|----|------------|---------|-------------------------------|
| 0 | `O` | *(rỗng)* | không có dấu câu sau từ này |
| 1 | `COMMA` | `,` | dấu phẩy |
| 2 | `PERIOD` | `.` | dấu chấm |
| 3 | `QUESTION` | `?` | dấu hỏi |

Nhãn `QMARK` của dữ liệu gốc được ánh xạ thành `QUESTION`. Không có nhãn nào
khác được tạo thêm.

```
bạn  đã  hoàn  thành  bài  tập  chưa      ngày  mai  chúng  ta  nộp  bài
 O   O    O      O     O    O   QUESTION   O     O     O    O    O   PERIOD
```

## 3. Dataset

| | |
|---|---|
| Nguồn | [JointCapPunc](https://github.com/ductho9799/JointCapPunc) |
| Commit (ghim & xác minh) | `ee258cae0e95e64245428d59ebbb030280fcebec` |
| License | Apache-2.0 (`JOINTCAPPUNC_LICENSE.txt`) |
| Định dạng gốc | `word<TAB>capitalization_label<TAB>punctuation_label` |
| Lĩnh vực | hội thoại tư vấn y tế tiếng Việt |

### Split chính thức được giữ nguyên tuyệt đối

| file gốc | split | dùng khi nào |
|---|---|---|
| `train.txt` | `train` | huấn luyện |
| `dev.txt` | `validation` | chọn epoch tốt nhất **và** chọn winner |
| `test.txt` | `test` | **một lần duy nhất**, sau khi khoá winner |

### Dữ liệu sau xử lý

| split | examples | tokens | mean len | max len |
|---|---:|---:|---:|---:|
| train | 38,066 | 4,940,478 | 129.8 | 150 |
| validation | 15,227 | 1,977,406 | 129.9 | 150 |
| test | 22,832 | 2,968,815 | 130.0 | 150 |
| *all_labeled* | *76,125* | *9,886,699* | | *(chỉ audit/thống kê)* |

### Phân bố nhãn (tỉ lệ token)

| split | `O` | `COMMA` | `PERIOD` | `QUESTION` |
|---|---:|---:|---:|---:|
| train | 91.101% | 4.616% | 3.728% | 0.556% |
| validation | 91.058% | 4.643% | 3.743% | 0.556% |
| test | 91.042% | 4.635% | 3.759% | 0.563% |

Mất cân bằng nặng — đây là lý do E3/E4 tồn tại, và là lý do metric chính loại
`O` ra ngoài.

## 4. Data Pipeline

```bash
python scripts/run_data_pipeline.py
```

```
download ──► verify commit ──► parse ──► normalize ──► map labels ──► chunk
    ──► deduplicate ──► write JSONL ──► validate ──► statistics
    ──► class weights ──► hash ──► manifest ──► human-review samples
```

1. **download + verify** — clone, checkout đúng commit, xác minh
   `git rev-parse HEAD`, hash 3 file gốc, chép LICENSE, ghi manifest.
2. **parse** — đọc nghiêm ngặt 3 cột; dòng sai định dạng **raise kèm
   `file:line`**, không bao giờ bị bỏ qua âm thầm.
3. **normalize** — Unicode NFC, gom khoảng trắng, lowercase.
   **Dấu thanh luôn được giữ**: `tiếng việt` không bao giờ thành `tieng viet`.
4. **map labels** — `QMARK → QUESTION`; nhãn lạ raise, không gom vào `O`.
5. **chunk** — tối đa 150 từ/example, ưu tiên cắt tại `PERIOD`/`QUESTION`
   (98.8% example kết thúc đúng ranh giới câu). Bất biến được assert: không
   mất, không lặp, không đảo thứ tự token.
6. **deduplicate** — bỏ trùng chính xác **trong từng split** (train 63,
   validation 27, test 32). Không chuyển example giữa các split.
7. **validate** — đọc lại file đã ghi, kiểm tra mọi bất biến.
8. **statistics** — thống kê, class weight (chỉ từ train), hash, 100 mẫu review.

Chạy lại cho hash **giống hệt** — đã kiểm chứng bằng cách chạy pipeline hai lần
và so `content_sha256`.

### Kiểm tra dữ liệu

`outputs/data/validation_report.json`: **PASS, 0 lỗi, 3 cảnh báo**.

Cảnh báo là **trùng text giữa các split**: train↔validation 53 (0.348%),
train↔test 73 (0.320%), validation↔test 52 (0.342%). Đây là đặc tính có thật
của corpus hội thoại lớn (`vâng ạ.`, `cảm ơn bác sĩ.` xuất hiện tự nhiên ở
nhiều split). Dự án **báo cáo chứ không xoá** — split chính thức phải giữ
nguyên như đã công bố; tự ý "sửa" sẽ làm benchmark không còn so sánh được.

### Class weights (chỉ tính từ TRAIN)

| nhãn | count | tỉ lệ | inverse (E3) | sqrt-inverse (E4) |
|---|---:|---:|---:|---:|
| `O` | 4,500,812 | 91.101% | 0.2744 | 0.5239 |
| `COMMA` | 228,032 | 4.616% | 5.4164 | 2.3273 |
| `PERIOD` | 184,186 | 3.728% | 6.7058 | 2.5896 |
| `QUESTION` | 27,448 | 0.556% | 44.9985 | 6.7081 |

`load_class_weights()` **raise** nếu `source_split != "train"`.

### Hash dữ liệu (`content_sha256`)

```
train        138983bc2761546f13c9520d5339b2ea10e588b756055fe85e895f5b6ce81e8e
validation   874133954b31fbfd73ccea2f84c69b15d106d4e8bc94f6f26ffcd4d2bdcc1fad
test         960cf879be7ac275ce16da846c9d37fd70838ada2c5b56ea2ab7428ccd15f3e1
```

Cả bốn experiment ghi các hash này vào artifact và metadata checkpoint, nên
luôn truy ngược được checkpoint ↔ dữ liệu.

## 5. Repository Structure

```
NLPgrSix7/
├── README.md  pytest.ini  .gitignore
├── requirements.txt  requirements-lock.txt
├── JOINTCAPPUNC_LICENSE.txt
│
├── configs/          data.yaml + e1..e4 yaml
├── data/
│   ├── raw/JointCapPunc/        (checkout đã ghim, gitignored)
│   └── processed/               train/validation/test/all_labeled .jsonl
│
├── scripts/
│   ├── _bootstrap.py                          import src từ mọi nơi
│   ├── download_data.py                       stage 1
│   ├── prepare_data.py                        stage 2
│   ├── validate_data.py                       stage 3
│   ├── compute_statistics.py                  stage 4
│   ├── run_data_pipeline.py                   chạy cả 4 stage
│   ├── smoke_test.py                          kiểm tra code training chạy được
│   ├── check_experiments.py                   trạng thái E1–E4
│   ├── package_release.py                     đóng gói .zip để nộp bài
│   ├── generate_deliverables.py               tổng hợp các artifact bàn giao
│   ├── build_evaluation_report.py             dựng report PDF từ artifact
│   ├── verify_experiments.py                  Phase 2 step 1
│   ├── compute_unweighted_validation_loss.py  Phase 2 step 2
│   └── demo_inference.py                      CLI inference
│
├── src/
│   ├── data/         constants, parser, normalization, chunking, dedup,
│   │                 schema, validation, statistics, dataset
│   ├── models/       bilstm, phobert, factory
│   ├── training/     seed, losses, optimizer, scheduler, trainer_base,
│   │                 trainer_bilstm, trainer_phobert, checkpointing, artifacts
│   ├── evaluation/   metrics, evaluator, loaders, selection, baselines,
│   │                 error_analysis
│   ├── inference/    tokenizer, predictor, reconstruction, service
│   └── utils/        io, hashing, logging_utils, environment
│
├── notebooks/
│   ├── 01_E1_BiLSTM.ipynb                    ─┐
│   ├── 02_E2_PhoBERT_NoWeight.ipynb           │ training (train+validation)
│   ├── 03_E3_PhoBERT_InverseWeight.ipynb      │ KHÔNG đọc test
│   ├── 04_E4_PhoBERT_SqrtInverse.ipynb       ─┘
│   ├── 05_Model_Comparison_Selection.ipynb    chọn + khoá winner (validation)
│   ├── 06_Final_Test_Evaluation.ipynb         official test, một lần
│   ├── 07_Posthoc_Test_Comparison.ipynb       mô tả, không đổi winner
│   └── 08_Checkpoint_F1_Visualization.ipynb   kiểm chứng/biểu đồ checkpoint tùy chọn
│
├── app/              app.py (Streamlit) + examples.py
├── outputs/
│   ├── data/         artifact của data pipeline
│   ├── checkpoints/  E1..E4
│   ├── experiments/  E1..E4
│   ├── evaluation/   selection, test, post-hoc, error analysis, demo
│   └── figures/      biểu đồ so sánh + confusion matrix
└── tests/            14 file test + conftest.py, 262 test
```

## 6. Data Pipeline

| file | nhiệm vụ |
|---|---|
| `src/data/constants.py` | nguồn sự thật: nhãn, đường dẫn, commit ghim, revision PhoBERT, seed |
| `src/data/jointcappunc_parser.py` | đọc TSV 3 cột strict, lỗi kèm `file:line`, blank line = ranh giới document |
| `src/data/normalization.py` | NFC + whitespace + lowercase, `QMARK→QUESTION`, **giữ dấu thanh** |
| `src/data/chunking.py` | tách câu → gom ≤150 từ → hard-cut khi buộc phải + assert bất biến |
| `src/data/deduplication.py` | bỏ trùng trong từng split, báo cáo số lượng |
| `src/data/schema.py` | schema example + validator |
| `src/data/validation.py` | mọi check ERROR/WARNING trên file đã ghi |
| `src/data/statistics.py` | thống kê, class weight, hash, human-review |
| `src/data/dataset.py` | `Vocabulary`, `PhoBERTEncoder`, **align word↔subword**, torch Dataset |

## 7. Model / Training / Evaluation / Inference

| file | nhiệm vụ |
|---|---|
| `src/models/bilstm.py` | BiLSTM tagger + `pack_padded_sequence` |
| `src/models/phobert.py` | wrapper `RobertaForTokenClassification`, ghim revision |
| `src/models/factory.py` | `build_model`, `describe_model`, `load_model_from_checkpoint` |
| `src/training/seed.py` | seed random/numpy/torch/CUDA/DataLoader |
| `src/training/losses.py` | loss factory `none`/`inverse`/`sqrt_inverse`, chặn weight ngoài train |
| `src/training/optimizer.py` | AdamW + nhóm no-decay cho bias & LayerNorm |
| `src/training/scheduler.py` | ReduceLROnPlateau (E1), linear warmup+decay (E2–E4) |
| `src/training/trainer_base.py` | vòng lặp chung, validation mỗi epoch, chọn best |
| `src/training/trainer_bilstm.py` | vòng lặp E1 |
| `src/training/trainer_phobert.py` | vòng lặp E2–E4: grad accumulation + fp16 |
| `src/training/checkpointing.py` | chỉ giữ best checkpoint + metadata đầy đủ |
| `src/training/artifacts.py` | mọi file trong `outputs/experiments/<E>/` |
| `src/evaluation/metrics.py` | confusion matrix, per-class P/R/F1, **Punctuation Macro-F1** |
| `src/evaluation/evaluator.py` | vòng đánh giá chung, gộp dự đoán theo từ |
| `src/evaluation/loaders.py` | dựng DataLoader khớp với kiến trúc của checkpoint |
| `src/evaluation/selection.py` | chọn winner **chỉ từ validation**, khoá winner |
| `src/evaluation/baselines.py` | B0 (all-O) và B1 (heuristic từ khoá) |
| `src/evaluation/error_analysis.py` | cặp nhầm lẫn + ví dụ thật kèm ngữ cảnh |
| `src/inference/tokenizer.py` | text → từ, bảo vệ URL/email/số |
| `src/inference/predictor.py` | nạp winner, dự đoán, cửa sổ hoá văn bản dài |
| `src/inference/reconstruction.py` | từ + nhãn → văn bản có dấu câu |
| `src/inference/service.py` | singleton có cache + xử lý lỗi có cấu trúc |

## 8–11. Bốn thí nghiệm

| | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| mô hình | BiLSTM (from scratch) | PhoBERT-base-v2 | PhoBERT-base-v2 | PhoBERT-base-v2 |
| tiền huấn luyện | ❌ | ✅ | ✅ | ✅ |
| class weight | không | không | inverse | sqrt-inverse |
| epochs | 12 | 5 | 5 | 5 |
| batch hiệu dụng | 128 | 4×4 ≈ 16 | 4×4 ≈ 16 | 4×4 ≈ 16 |
| learning rate | 1e-3 | 3e-5 | 3e-5 | 3e-5 |
| weight decay | 1e-4 | 1e-2 | 1e-2 | 1e-2 |
| scheduler | ReduceLROnPlateau | linear warmup 10% | linear warmup 10% | linear warmup 10% |
| fp16 | ❌ | ✅ | ✅ | ✅ |
| seed | 42 | 42 | 42 | 42 |
| best epoch | 11 | 5 | 5 | 5 |
| thời gian thực tế | 203 s | 2,467 s | 2,466 s | 20,617 s* |

\* E4 chạy chậm bất thường vì lúc đó ba kernel notebook trước chưa được tắt và
GPU 8 GB bị đầy VRAM. **Không phải đặc tính của sqrt-inverse weighting** — ba
thí nghiệm PhoBERT có khối lượng tính toán y hệt nhau. Chi tiết:
`outputs/experiments/E4/TIMING_NOTE.md`.

**Kiến trúc E1.** `Embedding(31,615 → 128) → BiLSTM(128/chiều) →
Dropout(0.30) → Linear(256 → 4)`, vocabulary **chỉ xây từ train**, `<PAD>=0`,
`<UNK>=1`.

**Kiến trúc E2/E3/E4.** `vinai/phobert-base-v2` ghim revision
`fb76b7e1f77fa19bc4870e2ad956876f7c81c53f`, `RobertaForTokenClassification`,
`num_labels=4`, `max_length=192` subword. Ba thí nghiệm **chỉ khác nhau một
dòng config** (`loss.weight_mode`), nên mọi chênh lệch quy được về class weight.

## 12. Training Protocol

* validation chạy **sau mỗi epoch** trên toàn bộ `validation.jsonl`;
* checkpoint tốt nhất chọn theo **validation Punctuation Macro-F1**;
* chỉ giữ một checkpoint mỗi experiment;
* metadata checkpoint ghi: experiment id, model + revision, label map, config
  đầy đủ, best epoch, điểm validation, hash dữ liệu, seed, phiên bản
  Python/PyTorch/Transformers/CUDA và tên GPU;
* **không notebook training nào nạp `test.jsonl`** (có guard trong notebook và
  test tự động quét mã nguồn).

### Word ↔ subword alignment (phần tinh tế nhất)

Dấu câu nằm **sau từ**, nên nhãn đặt tại **subword cuối cùng** của từ:

```
từ:        [ hoàn ]  [ thành ]  [   chưa   ]
subword:    ho@@ àn   thành      ch@@  ưa
nhãn:       -100  O   PERIOD     -100  QUESTION
```

Mọi vị trí khác (`<s>`, `</s>`, subword không phải cuối, padding) mang
`IGNORE_INDEX = -100`. Chuỗi dài hơn 192 subword được **cắt thành nhiều cửa sổ
tại ranh giới từ**, không bao giờ truncate. Bất biến: mỗi từ được giám sát đúng
một lần.

## 13. Validation Model Selection

```bash
python scripts/verify_experiments.py
python scripts/compute_unweighted_validation_loss.py

```

**Metric chính:** validation Punctuation Macro-F1 = `mean(F1_COMMA, F1_PERIOD,
F1_QUESTION)`. Accuracy **không** được dùng: model đoán `O` cho mọi từ đạt
0.9104 accuracy mà PUNCT-F1 = 0.0000 (chính là baseline B0 bên dưới).

**Tie-break:** validation **unweighted** loss. Cần cẩn thận ở đây — mỗi
experiment ghi loss theo hàm loss của riêng nó (E3 inverse-weighted, E4
sqrt-weighted), ba con số đó **không so sánh được**. Vì vậy
`scripts/compute_unweighted_validation_loss.py` nạp lại cả bốn checkpoint và
tính lại loss bằng **một CrossEntropyLoss không trọng số dùng chung**. Script
này đồng thời xác nhận cả bốn checkpoint **tái tạo đúng** điểm validation đã
báo cáo (sai lệch tối đa 3.1e-05).

### Kết quả validation

| rank | exp | weight_mode | best epoch | loss (unweighted) | accuracy | macro-F1 | **PUNCT-F1** | F1 COMMA | F1 PERIOD | F1 QUESTION |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **E2** | none | 5 | 0.090323 | 0.966725 | 0.830614 | **0.778718** | 0.726973 | 0.810835 | 0.798345 |
| 2 | E4 | sqrt_inverse | 5 | 0.114305 | 0.957236 | 0.804645 | 0.745676 | 0.700309 | 0.791063 | 0.745657 |
| 3 | E3 | inverse | 5 | 0.185980 | 0.935833 | 0.763878 | 0.695324 | 0.612241 | 0.758988 | 0.714742 |
| 4 | E1 | none | 11 | 0.158479 | 0.944757 | 0.714699 | 0.628241 | 0.496436 | 0.686458 | 0.701829 |

### Ba câu hỏi nghiên cứu, trả lời bằng số liệu

1. **Tiền huấn luyện tiếng Việt có đáng không?** Có, rất rõ.
   E2 vs E1: **+0.1505 PUNCT-F1 (+24.0%)**. Riêng `COMMA` — lớp khó nhất —
   tăng từ 0.496 lên 0.727.
2. **Class weight có giúp không?** **Không — nó làm hại.**
   E3 (inverse) vs E2: **−0.0834 (−10.7%)**. E4 (sqrt-inverse) vs E2:
   **−0.0330 (−4.2%)**. Trọng số càng mạnh, kết quả càng tệ.
3. **Chênh lệch có đủ lớn để kết luận không?** Có. Khoảng cách E2→E4 là 0.033,
   lớn hơn nhiều so với mức nhiễu (chênh validation↔test của E2 chỉ 0.0024).

**Vì sao class weight lại làm hại?** Trọng số kéo mô hình về phía lớp hiếm nên
recall tăng, nhưng precision tụt mạnh hơn: mô hình rắc dấu câu vào những chỗ
không có. PhoBERT vốn đã học đủ tốt để tự xử lý mất cân bằng, nên can thiệp
thêm chỉ phá vỡ cân bằng precision/recall. Đây là một **kết quả âm** và được
báo cáo đúng như vậy.

## 14. Winner

`outputs/evaluation/model_selection.json`:

```json
{
  "selection_split": "validation",
  "selection_metric": "punctuation_macro_f1",
  "tie_breaker": "unweighted_validation_loss",
  "winner": "E2",
  "test_was_used_for_selection": false,
  "winner_locked": true
}
```

* Winner: **E2** — `vinai/phobert-base-v2`, không class weight, best epoch 5
* Runner-up: E4, cách biệt **0.0330** → tie-break không cần dùng
* Checkpoint: `outputs/checkpoints/E2/`

**Winner bị khoá.** `load_locked_winner()` raise nếu `winner_locked` không phải
`true` hoặc nếu `test_was_used_for_selection` không phải `false`. Mọi thành
phần phía sau (notebook 06/07, inference, UI) đều đọc winner từ file này —
không nơi nào hardcode `"E2"`.

Rò rỉ test vào bước chọn model là **bất khả thi về mặt cấu trúc**:
`select_winner()` chỉ nhận `ValidationCandidate`, một frozen dataclass mà toàn
bộ trường là số liệu validation. Truyền dict có `test_punctuation_macro_f1` vào
sẽ raise `TypeError` (có test kiểm tra điều này).

## 15. Official Test

```bash

```

22,832 example / 2,968,815 từ. Đây là lần đầu tiên và duy nhất `test.jsonl`
được dùng trong dự án.

| metric | test | validation | chênh |
|---|---:|---:|---:|
| **Punctuation Macro-F1** | **0.776322** | 0.778718 | −0.002395 |
| Accuracy | 0.966265 | 0.966725 | −0.000460 |
| Macro-F1 (4 lớp) | 0.828767 | 0.830614 | −0.001847 |

Chênh lệch chỉ 0.24% → mô hình tổng quát hoá tốt và việc chọn model bằng
validation là đáng tin.

### Per-class trên test

| nhãn | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| `O` | 0.9861 | 0.9861 | 0.9861 | 2,702,879 |
| `COMMA` | 0.7407 | 0.7054 | 0.7226 | 137,612 |
| `PERIOD` | 0.7948 | 0.8240 | 0.8091 | 111,606 |
| `QUESTION` | 0.7484 | 0.8528 | 0.7972 | 16,718 |

`COMMA` là lớp khó nhất — hợp lý, vì vị trí dấu phẩy trong tiếng Việt phụ thuộc
văn phong và thường không có đáp án duy nhất. `QUESTION`, dù chỉ chiếm 0.56% dữ
liệu, lại đạt F1 0.797 nhờ các từ khoá cuối câu rất đặc trưng (`không`, `chưa`,
`ạ`, `hả`).

## 16. Optional Post-hoc Comparison

`notebooks/07_Posthoc_Test_Comparison.ipynb` — **mô tả, không đổi winner.**
Notebook có guard: từ chối chạy nếu winner chưa bị khoá.

| test rank | hệ thống | loại | weight_mode | validation PUNCT-F1 | **test PUNCT-F1** | test accuracy |
|---|---|---|---|---:|---:|---:|
| 1 | **E2** | trained | none | 0.778718 | **0.776322** | 0.966265 |
| 2 | E4 | trained | sqrt_inverse | 0.745676 | 0.743845 | 0.956761 |
| 3 | E3 | trained | inverse | 0.695324 | 0.694776 | 0.935458 |
| 4 | E1 | trained | none | 0.628241 | 0.624629 | 0.944153 |
| 5 | B1 | baseline | – | – | 0.166006 | 0.831232 |
| 6 | B0 | baseline | – | – | 0.000000 | **0.910424** |

**Thứ tự trên validation và trên test giống hệt nhau** (E2 > E4 > E3 > E1) —
bằng chứng mạnh cho thấy giao thức chọn model là đáng tin. Winner cũng dẫn đầu
trên test, nên không có xung đột cần xử lý.

**B0 là minh hoạ sống động nhất của dự án:** đoán `O` cho mọi từ đạt **91.04%
accuracy** nhưng **0.0000 PUNCT-F1**. Nếu chọn model bằng accuracy, B0 sẽ xếp
trên cả E3 và E1.

### Baselines

* **B0** — đoán `O` cho mọi từ.
* **B1** — heuristic từ khoá tiếng Việt, thuần luật, **không dùng dữ liệu huấn
  luyện**: đặt `QUESTION` sau các từ nghi vấn cuối câu (`không`, `chưa`, `hả`,
  `à`…), `PERIOD` sau các từ kết câu (`rồi`, `nhé`, `ạ`…) hoặc khi câu quá 25
  từ, `COMMA` trước các liên từ mở mệnh đề (`nhưng`, `nếu`, `vì`…). Luật đầy đủ
  in ra trong notebook 07 và trong `src/evaluation/baselines.py`.

Baseline **không** tham gia chọn winner; chúng chỉ định nghĩa "sàn" của bài
toán. E2 vượt B1 gấp **4.7 lần** về PUNCT-F1.

## 17. Metrics

**Metric chính — Punctuation Macro-F1:**

```
PUNCT-F1 = ( F1_COMMA + F1_PERIOD + F1_QUESTION ) / 3
```

`O` bị loại vì nó chiếm 91% dữ liệu (xem B0). Macro chứ không micro, vì
`QUESTION` chỉ chiếm 0.56% và sẽ vô hình trong micro-average.

Các chỉ số khác (accuracy, macro-F1 4 lớp, micro-F1, P/R/F1 từng lớp) đều được
ghi lại đầy đủ nhưng **không** dùng để ra quyết định.

## 18. Error Analysis

Phân tích trên 4,000 example test (520,335 từ), error rate **3.38%**.

| gold → pred | count | % tổng lỗi | nghĩa thực tế |
|---|---:|---:|---|
| `COMMA → O` | 4,844 | 27.5% | bỏ sót dấu phẩy — hai mệnh đề dính nhau |
| `O → COMMA` | 4,342 | 24.7% | thêm dấu phẩy thừa |
| `COMMA → PERIOD` | 2,134 | 12.1% | cắt câu quá mạnh |
| `O → PERIOD` | 1,967 | 11.2% | chấm câu quá sớm, chẻ đôi một câu |
| `PERIOD → COMMA` | 1,567 | 8.9% | kết câu quá yếu, thành câu ghép |
| `PERIOD → O` | 1,466 | 8.3% | bỏ sót dấu chấm (run-on) |
| `O → QUESTION` | 343 | 1.9% | thêm dấu hỏi thừa |
| `PERIOD → QUESTION` | 327 | 1.9% | nhầm câu kể thành câu hỏi |

**Nhận định chính:** hơn **52%** số lỗi liên quan tới `COMMA` (bỏ sót hoặc thêm
thừa). Điều này hợp lý về mặt ngôn ngữ — vị trí dấu phẩy trong tiếng Việt phần
lớn là lựa chọn văn phong, nhiều câu có nhiều cách chấm phẩy đều đúng. Nhóm lỗi
ranh giới câu (`PERIOD ↔ O`, `PERIOD ↔ COMMA`) chiếm ~29% và mới là nhóm ảnh
hưởng thực sự tới khả năng đọc. Lỗi liên quan `QUESTION` chỉ ~4%.

Ví dụ thật kèm ngữ cảnh: `outputs/evaluation/final_error_analysis.csv`.

## 19. Inference

```python
from src.inference import PunctuationRestorationPredictor

predictor = PunctuationRestorationPredictor.from_selected_model()
result = predictor.restore("bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài")
print(result.restored_text)

```

```bash
python scripts/demo_inference.py
python scripts/demo_inference.py --interactive
python scripts/demo_inference.py --text "..."
python scripts/demo_inference.py --file in.txt
python scripts/demo_inference.py --save
```

### Kết quả thật (winner E2, GPU)

| input | output |
|---|---|
| `bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài` | Bạn đã hoàn thành bài tập chưa? Ngày mai chúng ta nộp bài. |
| `chào bác sĩ em bị đau bụng mấy hôm nay rồi em có nên đi khám không em cảm ơn bác sĩ nhiều` | Chào bác sĩ, em bị đau bụng mấy hôm nay rồi. Em có nên đi khám không? Em cảm ơn bác sĩ nhiều. |
| `tôi muốn đi chơi nhưng trời đang mưa rất to nếu chiều nay tạnh thì chúng ta sẽ đi công viên` | Tôi muốn đi chơi nhưng trời đang mưa rất to, nếu chiều nay tạnh thì chúng ta sẽ đi công viên. |
| `em bị sốt 38.5 độ từ tối qua uống thuốc hạ sốt rồi mà chưa đỡ em phải làm sao ạ` | Em bị sốt 38.5 độ từ tối qua, uống thuốc hạ sốt rồi mà chưa đỡ, em phải làm sao ạ? |

Toàn bộ 8 ví dụ: `outputs/evaluation/inference_demo.json`.

**Độ trễ demo** (RTX 4060 Laptop, fp16, sau warm-up 254 ms): 10–63 từ →
min 7.7 ms, median 8.2 ms, max 21.8 ms. Đây là *demo latency*, **không phải
benchmark production** (không lặp lại nhiều lần, không nghiên cứu batching,
không báo cáo phương sai).

### Chi tiết kỹ thuật

* **Không hardcode winner.** Predictor đọc `model_selection.json`, tra
  `checkpoint_metadata.json` để biết kiến trúc, rồi nạp BiLSTM hoặc PhoBERT
  tương ứng. Nếu train lại và E1 thắng, code không cần sửa một dòng.
* **Tokenizer bảo vệ** URL, email, số thập phân (`38.5`, `2.500.000`, `12:30`,
  `3/4`), từ ghép nối gạch (`x-quang`, `covid-19`).
* **Văn bản dài không bị cắt cụt.** PhoBERT chia thành nhiều cửa sổ 192 subword
  tại ranh giới từ rồi ghép dự đoán lại theo chỉ số từ; BiLSTM chia theo 150
  từ. Bất biến `len(labels) == len(words)` được assert. Vượt giới hạn cấu hình
  thì **raise**, không truncate âm thầm.
* **Viết hoa là rule-based** (đầu văn bản, sau `PERIOD`/`QUESTION`). Dự án
  **không** có mô hình capitalization — tên riêng sẽ không được viết hoa.

## 20. Streamlit UI

```bash
streamlit run app/app.py
```

* đọc winner từ `model_selection.json` — **không hardcode**;
* `@st.cache_resource` nạp model một lần cho cả phiên;
* ô nhập text + nút **Khôi phục dấu câu** + nút **Xoá**;
* hiển thị song song input / output, cùng số từ, số câu, số dấu phẩy, số dấu
  hỏi và độ trễ;
* sidebar: experiment thắng cuộc, model, kiến trúc, thiết bị, cách chọn winner,
  điểm validation và điểm test, danh sách giới hạn đã biết;
* 8 ví dụ mẫu bấm một nút là điền vào ô nhập (**không** lưu sẵn output nào —
  mọi kết quả đều do model chạy thật);
* xử lý lỗi: input rỗng, thiếu `model_selection.json`, checkpoint hỏng, input
  quá dài, hết VRAM, lỗi bất ngờ — tất cả hiện thông báo, không trang trắng.

UI được kiểm thử **headless bằng `streamlit.testing.v1.AppTest`**: chạy thật
app, nạp winner, bấm nút, xác nhận có dấu câu trong kết quả.

## 21. How to Run From Scratch

```bash

git clone <repo-url> && cd NLPgrSix7
python -m venv .venv && .venv\Scripts\activate


pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt


python -c "import torch; print(torch.__version__, torch.cuda.is_available())"





python scripts/run_data_pipeline.py


python -m compileall src scripts app
python -m pytest -q






python scripts/check_experiments.py


python scripts/verify_experiments.py
python scripts/compute_unweighted_validation_loss.py









python scripts/demo_inference.py --save


streamlit run app/app.py
```

> ⚠️ **Tắt kernel notebook trước khi mở notebook tiếp theo.** GPU 8 GB không
> chứa được hai PhoBERT cùng lúc; để kernel cũ sống làm E4 chậm gấp 8 lần
> (xem `outputs/experiments/E4/TIMING_NOTE.md`).

### Môi trường đã kiểm chứng

Toàn bộ **262 test pass** trên cấu hình hiện tại; 261 test gốc đã được kiểm chứng trên hai cấu hình dưới đây:

| | A (venv của dự án) | B (Python global) |
|---|---|---|
| Python | **3.12.4** | 3.13.2 |
| torch | 2.13.0+cu126 | 2.13.0+cu126 |
| transformers | 5.14.1 | 5.14.1 |
| numpy / pandas | 2.5.1 / 3.0.5 | 2.2.5 / 2.2.3 |
| streamlit | 1.61.1 | 1.61.1 |
| GPU | RTX 4060 Laptop 8GB · CUDA 12.6 | như A |

Code chạy được với cả pandas 2.x và 3.x. `requirements-lock.txt` khoá chính xác
cấu hình A.

## 21b. Đóng gói để nộp bài

Cả repository nặng ~2.0 GB (checkpoint 1.6 GB + dữ liệu 284 MB), không nộp
nguyên trạng được. Dùng:

```bash
python scripts/package_release.py --profile submission
python scripts/package_release.py --profile code
python scripts/package_release.py --profile full
python scripts/package_release.py --profile submission --dry-run
```

| profile | nội dung | zip |
|---|---|---:|
| `code` | code + notebook (kèm output) + **toàn bộ báo cáo & biểu đồ**. Không data, không trọng số. | ~2 MB |
| `submission` | như `code` + **checkpoint winner** (đọc từ `model_selection.json`, không hardcode) → chạy được inference + UI ngay | **~478 MB** |
| `full` | thêm cả 4 checkpoint và `data/processed/` | ~1.7 GB |

Mọi profile đều loại `.venv/`, `.git/`, `__pycache__/`, `data/raw/` (clone lại
được bằng một lệnh) và `outputs/smoke/` (kết quả smoke test, không phải kết quả
thật).

Zip luôn kèm `PACKAGE_MANIFEST.json`: profile, số file, dung lượng theo thư mục,
winner, điểm test, hash dữ liệu, và các lệnh khôi phục — để người chấm biết
chính xác họ đang cầm thứ gì.

## 22. How to Reproduce Training

Mọi tham số nằm trong `configs/`; notebook không hardcode số nào. Muốn đổi thí
nghiệm thì sửa YAML, không sửa notebook.

* seed 42 cho `random`, `numpy`, `torch`, CUDA, DataLoader;
* dữ liệu ghim bằng `content_sha256`, ghi vào metadata checkpoint;
* PhoBERT ghim revision `fb76b7e1f77fa19bc4870e2ad956876f7c81c53f`;
* hết VRAM: giảm `train_batch_size` xuống 2 và tăng
  `gradient_accumulation_steps` lên 8 (batch hiệu dụng không đổi).

**Lưu ý về tính tất định:** trên GPU, cuDNN và các phép reduce không tất định
có thể làm hai lần chạy khác nhau ở vài chữ số cuối. Bật
`training.deterministic: true` trong config để giảm điều này, đổi lại tốc độ
chậm hơn.

## 23. Artifact Structure

```
outputs/
├── data/                                    (data pipeline)
│   ├── data_source_manifest.json            nguồn + commit đã xác minh
│   ├── preparation_report.json              báo cáo từng bước
│   ├── validation_report.json               PASS, 0 lỗi, 3 cảnh báo
│   ├── data_statistics.json / .csv
│   ├── class_weights.json                   chỉ từ train
│   ├── data_hashes.json
│   ├── human_review_samples.csv             100 mẫu (is_correct để trống)
│   └── pipeline_receipt.json
│
├── checkpoints/E1..E4/                      chỉ giữ best checkpoint
│   ├── model.safetensors  (PhoBERT)  |  model.pt + vocabulary.json  (BiLSTM)
│   ├── config.json + tokenizer files (PhoBERT)
│   └── checkpoint_metadata.json
│
├── experiments/E1..E4/
│   ├── config.json, training_history.csv
│   ├── best_validation_metrics.json
│   ├── per_class_validation_metrics.csv
│   ├── validation_confusion_matrix.csv
│   ├── validation_sample_predictions.csv
│   ├── training_curves.png
│   ├── environment.json, data_hashes.json
│   └── experiment_summary.json              status = COMPLETED
│
├── evaluation/
│   ├── training_verification.json           kiểm tra 4 lần training
│   ├── validation_unweighted_loss.json      loss so sánh được + tái tạo điểm
│   ├── validation_model_comparison.csv/json
│   ├── model_selection.json                 WINNER, LOCKED
│   ├── final_test_results.json              official test
│   ├── final_test_per_class.csv
│   ├── final_test_confusion_matrix.csv
│   ├── final_error_analysis.csv / .json
│   ├── posthoc_test_model_comparison.csv/json
│   ├── inference_demo.json
│   └── final_report.json                    tổng hợp toàn dự án
│
└── figures/
    ├── validation_model_comparison.png
    ├── validation_per_class_f1_comparison.png
    ├── final_confusion_matrix.png
    ├── final_per_class_f1.png
    └── posthoc_test_model_comparison.png
```

## 24. Limitations

1. **Miền dữ liệu hẹp.** Corpus là hội thoại tư vấn y tế. Kết quả trên tin tức,
   văn bản pháp luật hay khẩu ngữ đời thường sẽ thấp hơn con số 0.7763.
2. **Chỉ 4 nhãn.** Không có `!`, `:`, `;`, `...`, ngoặc kép, ngoặc đơn.
3. **Không có mô hình viết hoa.** Viết hoa là rule-based (đầu câu, sau `.`/`?`).
   Tên riêng không được viết hoa. JointCapPunc *có* nhãn capitalization nhưng
   dự án này cố ý không dùng.
4. **Dấu phẩy vốn mơ hồ.** Hơn 52% lỗi liên quan tới `COMMA`; nhiều trường hợp
   "sai" thực ra là một cách chấm phẩy hợp lệ khác.
5. **Trùng lặp giữa các split ~0.32–0.35%.** Đặc tính của corpus gốc; split
   chính thức được giữ nguyên và hiện tượng này được báo cáo, không bị che.
6. **Một seed duy nhất.** Mỗi thí nghiệm chạy đúng một lần với seed 42. Không
   có khoảng tin cậy; kết luận "E2 > E4" dựa trên cách biệt 0.033, đủ lớn so
   với nhiễu quan sát được (0.0024), nhưng vẫn không thay thế được nhiều seed.
7. **Latency chưa phải benchmark.** Chỉ là demo trên vài câu ngắn.
8. **Chưa có audio/ASR.**
9. **E4 có thời gian huấn luyện không so sánh được** vì thiếu VRAM lúc chạy
   (xem `outputs/experiments/E4/TIMING_NOTE.md`). Điểm số không bị ảnh hưởng.

## 25. Future ASR Integration

Hướng mở rộng tự nhiên:

```
audio  →  ASR (Whisper / PhoWhisper / wav2vec2-vi)
       →  transcript thô, không dấu câu
       →  MÔ HÌNH NÀY
       →  văn bản có dấu câu, dễ đọc
```

Đây chính là ứng dụng thực tế điển hình: đầu ra ASR gần như luôn thiếu dấu câu,
và đọc một đoạn transcript dài không dấu câu rất mệt.

**Chưa có dòng code audio/ASR nào trong repository này** — không có stub, không
có mock. Phần trên là kế hoạch, không phải tính năng.

Việc cần làm nếu triển khai: chọn mô hình ASR tiếng Việt, chuẩn hoá đầu ra ASR
về đúng dạng token mà mô hình này mong đợi, đo lỗi tích luỹ (WER của ASR sẽ kéo
theo lỗi chấm câu), và đánh giá end-to-end trên dữ liệu nói thật — chứ không
chỉ nối hai mô hình rồi coi là xong.

---

## Test suite

```bash
python -m pytest -q
```

| file | kiểm tra |
|---|---|
| `test_schema.py` | schema example, ID, cap 150 từ, nhãn hợp lệ |
| `test_parser.py` | strict parsing, lỗi kèm `file:line`, CRLF/LF, `QMARK→QUESTION` |
| `test_normalization.py` | NFC, **giữ dấu thanh**, lowercase, ánh xạ nhãn |
| `test_chunking.py` | không mất/lặp/đảo token, cap 150, ưu tiên ranh giới câu |
| `test_alignment.py` | nhãn ở subword cuối, mỗi từ giám sát đúng 1 lần |
| `test_metrics.py` | Punctuation Macro-F1, loại `O`, ignore_index |
| `test_model_selection.py` | chọn bằng validation, **test metric không lọt vào được** |
| `test_checkpoint_loading.py` | nạp 4 checkpoint thật, metadata đủ, smoke inference |
| `test_reconstruction.py` | label→ký hiệu, viết hoa, tokenizer bảo vệ URL/số |
| `test_predictor.py` | 1 nhãn/1 từ, dispatch kiến trúc, winner thật |
| `test_long_text.py` | **không truncate âm thầm** ở mọi tầng |
| `test_inference_service.py` | cache model, mọi lỗi thành response có cấu trúc |
| `test_app_smoke.py` | UI render headless, bấm nút, khôi phục được text |

## License

Code dự án: mục đích học thuật.
Dataset JointCapPunc: **Apache-2.0** — xem `JOINTCAPPUNC_LICENSE.txt`.
`vinai/phobert-base-v2`: theo license của tác giả trên Hugging Face.
