# Khôi phục dấu câu tiếng Việt

Dự án khôi phục dấu câu cho văn bản tiếng Việt chưa có dấu câu và chưa viết
hoa. Bài toán được xây dựng dưới dạng gán nhãn theo từ: mỗi từ nhận một nhãn
`O`, `COMMA`, `PERIOD` hoặc `QUESTION`.

Ví dụ:

```text
Đầu vào:  bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài
Đầu ra:   Bạn đã hoàn thành bài tập chưa? Ngày mai chúng ta nộp bài.
```

Model được dùng cho inference là E2, dựa trên `vinai/phobert-base-v2`. Model
này được chọn bằng kết quả validation; tập test không tham gia quá trình chọn
model.

## Kết quả chính

| Chỉ số | Kết quả |
|---|---:|
| Model được chọn | E2 - PhoBERT, không dùng class weight |
| Validation Punctuation Macro-F1 | 0.7787 |
| Test Punctuation Macro-F1 | 0.7763 |
| Test accuracy | 0.9663 |
| Test macro-F1 trên bốn lớp | 0.8288 |

Punctuation Macro-F1 là trung bình F1 của ba lớp `COMMA`, `PERIOD` và
`QUESTION`. Lớp `O` không được tính vào chỉ số này vì chiếm hơn 91% số token.

Kết quả chi tiết được lưu trong:

```text
outputs/evaluation/model_selection.json
outputs/evaluation/final_test_results.json
outputs/evaluation/final_test_per_class.csv
```

## Dữ liệu

Dự án sử dụng bộ dữ liệu
[JointCapPunc](https://github.com/ductho9799/JointCapPunc), phiên bản tại commit:

```text
ee258cae0e95e64245428d59ebbb030280fcebec
```

Ba split gốc được giữ nguyên:

| File nguồn | Split sử dụng | Mục đích |
|---|---|---|
| `train.txt` | train | Huấn luyện |
| `dev.txt` | validation | Chọn checkpoint và model |
| `test.txt` | test | Đánh giá model sau khi đã chốt winner |

Sau tiền xử lý:

| Split | Số mẫu | Số token | Độ dài trung bình |
|---|---:|---:|---:|
| Train | 38,066 | 4,940,478 | 129.8 |
| Validation | 15,227 | 1,977,406 | 129.9 |
| Test | 22,832 | 2,968,815 | 130.0 |

Pipeline phát hiện một lượng nhỏ câu trùng giữa các split. Các trường hợp này
được báo cáo nhưng không bị xóa vì dự án giữ nguyên cách chia dữ liệu chính
thức của JointCapPunc. Báo cáo nằm tại
`outputs/data/validation_report.json`.

Thư mục `data/raw/` và các file JSONL lớn trong `data/processed/` không được
đưa lên Git. Có thể tải và tạo lại chúng bằng pipeline của dự án.

## Các thí nghiệm

Bốn cấu hình được huấn luyện độc lập với seed 42:

| Thí nghiệm | Kiến trúc | Class weight | Best epoch | Validation Punctuation Macro-F1 |
|---|---|---|---:|---:|
| E1 | BiLSTM | Không | 11 | 0.6282 |
| E2 | PhoBERT | Không | 5 | 0.7787 |
| E3 | PhoBERT | Inverse | 5 | 0.6953 |
| E4 | PhoBERT | Sqrt-inverse | 5 | 0.7457 |

E2 đạt kết quả validation cao nhất và được khóa trong
`outputs/evaluation/model_selection.json`. Sau thời điểm đó, các bước
inference và đánh giá test đều đọc model từ file lựa chọn này thay vì ghi cứng
tên E2 trong code.

Ba thí nghiệm PhoBERT dùng cùng kiến trúc và tham số huấn luyện, chỉ khác cách
đặt trọng số cho hàm loss. Trên bộ dữ liệu này, class weight làm giảm
Punctuation Macro-F1 so với cấu hình không trọng số.

## Cấu trúc dự án

```text
NLPgrSix7/
|-- app/                 WebUI Streamlit
|-- configs/             Cấu hình dữ liệu và bốn thí nghiệm
|-- data/                Dữ liệu gốc và dữ liệu sau xử lý
|-- notebooks/           Notebook huấn luyện và đánh giá
|-- outputs/             Metric, biểu đồ, checkpoint và artifact
|-- report/              Báo cáo PDF
|-- scripts/             Pipeline dữ liệu, kiểm tra và đóng gói
|-- slides/              Slide trình bày
|-- src/
|   |-- data/            Parser, chuẩn hóa, chunking và dataset
|   |-- models/          BiLSTM và PhoBERT
|   |-- training/        Loss, optimizer, trainer và checkpoint
|   |-- evaluation/      Metric, lựa chọn model và phân tích lỗi
|   `-- inference/       Tokenizer, predictor và dựng lại văn bản
|-- tests/               Kiểm thử tự động
|-- requirements.txt     Khoảng phiên bản dependency được hỗ trợ
`-- requirements-lock.txt  Phiên bản đã dùng để kiểm chứng dự án
```

## Cài đặt

Python 3.12 được khuyến nghị. GPU không bắt buộc; khi không có CUDA, inference
sẽ chạy trên CPU.

Trên PowerShell:

```powershell
cd D:\NLPgrSix7
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra PyTorch và CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`requirements-lock.txt` ghi lại môi trường đã dùng để kiểm chứng. Nếu cần tái
tạo đúng môi trường CUDA 12.6:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-lock.txt
```

## Checkpoint

Checkpoint không được đưa lên Git vì dung lượng lớn. Để chạy inference hoặc
WebUI, cần có tối thiểu:

```text
outputs/checkpoints/E2/
outputs/evaluation/model_selection.json
```

Checkpoint E2 phải chứa trọng số, cấu hình, tokenizer và
`checkpoint_metadata.json`. Nếu nhận checkpoint từ một gói bàn giao, đặt toàn
bộ thư mục E2 vào đúng đường dẫn trên.

## Chạy WebUI

Chạy tại thư mục gốc của dự án:

```powershell
python -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501
```

Sau đó mở:

```text
http://127.0.0.1:8501
```

Với `127.0.0.1`, chỉ máy đang chạy Streamlit truy cập được. Giữ terminal mở
trong thời gian sử dụng và nhấn `Ctrl+C` để dừng server.

Khi người dùng bấm khôi phục dấu câu, WebUI gọi một tiến trình inference riêng.
Tiến trình này nạp checkpoint được chỉ định bởi `model_selection.json`, dự đoán
nhãn cho từng từ rồi dựng lại văn bản. Cách tách tiến trình giúp WebUI tiếp tục
hoạt động nếu thư viện model gặp lỗi trong lúc suy luận.

## Chạy inference bằng terminal

Một câu trực tiếp:

```powershell
python scripts/demo_inference.py --text "hôm nay trời đẹp bạn có muốn đi dạo không"
```

Chế độ nhập liên tục:

```powershell
python scripts/demo_inference.py --interactive
```

Đọc nội dung từ file:

```powershell
python scripts/demo_inference.py --file input.txt
```

Có thể gọi predictor từ Python:

```python
from src.inference import PunctuationRestorationPredictor

predictor = PunctuationRestorationPredictor.from_selected_model()
result = predictor.restore(
    "bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài"
)
print(result.restored_text)
```

PhoBERT tự chia văn bản dài thành nhiều cửa sổ tại ranh giới từ. Kết quả của
các cửa sổ được ghép lại theo vị trí từ, không cắt bỏ phần cuối văn bản.

## Tạo lại dữ liệu

Chạy toàn bộ pipeline:

```powershell
python scripts/run_data_pipeline.py
```

Pipeline thực hiện các bước sau:

1. Clone JointCapPunc và chuyển về commit đã chỉ định.
2. Kiểm tra ba file dữ liệu nguồn.
3. Chuẩn hóa Unicode, khoảng trắng và nhãn dấu câu.
4. Chia văn bản thành mẫu tối đa 150 từ.
5. Loại bản ghi trùng trong từng split.
6. Ghi dữ liệu JSONL và chạy validation.
7. Tính thống kê, class weight và hash dữ liệu.

Có thể chạy riêng từng bước:

```powershell
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/validate_data.py
python scripts/compute_statistics.py
```

## Huấn luyện và đánh giá

Các notebook được chạy theo thứ tự:

```text
01_E1_BiLSTM.ipynb
02_E2_PhoBERT_NoWeight.ipynb
03_E3_PhoBERT_InverseWeight.ipynb
04_E4_PhoBERT_SqrtInverse.ipynb
05_Model_Comparison_Selection.ipynb
06_Final_Test_Evaluation.ipynb
07_Posthoc_Test_Comparison.ipynb
08_Checkpoint_F1_Visualization.ipynb
```

Bốn notebook đầu chỉ dùng train và validation. Notebook 05 so sánh các model
trên validation và khóa winner. Notebook 06 mới đánh giá winner trên test.
Notebook 07 chỉ phân tích sau đánh giá và không thay đổi model đã chọn.

Kiểm tra artifact của các thí nghiệm:

```powershell
python scripts/check_experiments.py
python scripts/verify_experiments.py
python scripts/compute_unweighted_validation_loss.py
```

Nếu huấn luyện PhoBERT trên GPU 8 GB, nên tắt kernel của notebook trước khi mở
notebook tiếp theo để giải phóng VRAM.

## Kiểm thử

Chạy toàn bộ test:

```powershell
python -m pytest -q
```

Kết quả tại lần kiểm tra gần nhất:

```text
262 passed
```

Bộ test bao phủ parser, chuẩn hóa, chunking, word-subword alignment, metric,
lựa chọn model, nạp checkpoint, inference văn bản dài và WebUI Streamlit.

Kiểm tra nhanh cú pháp:

```powershell
python -m compileall -q app src scripts tests
```

## Artifact và đóng gói

Các kết quả nhỏ trong `outputs/data/`, `outputs/experiments/`,
`outputs/evaluation/` và `outputs/figures/` có thể đưa lên Git. Hai thư mục sau
được bỏ qua:

```text
outputs/checkpoints/
outputs/smoke/
```

Tạo gói bàn giao:

```powershell
python scripts/package_release.py --profile code
python scripts/package_release.py --profile submission
python scripts/package_release.py --profile full
```

`code` không chứa checkpoint. `submission` chứa checkpoint của model được chọn.
`full` chứa cả bốn checkpoint và dữ liệu đã xử lý. Có thể xem trước danh sách
file mà không tạo ZIP:

```powershell
python scripts/package_release.py --profile submission --dry-run
```

## Giới hạn

- Dữ liệu chủ yếu là hội thoại tư vấn y tế; kết quả ở miền khác có thể thấp hơn.
- Model chỉ dự đoán dấu phẩy, dấu chấm và dấu hỏi.
- Viết hoa đầu câu được xử lý bằng quy tắc, không phải model capitalization.
- Tên riêng không được nhận diện để viết hoa.
- Dấu phẩy là lớp khó nhất và chiếm phần lớn lỗi dự đoán.
- Mỗi thí nghiệm mới chạy với một seed nên chưa có khoảng tin cậy qua nhiều lần chạy.
- Dự án hiện chỉ xử lý văn bản, chưa nhận âm thanh hoặc chạy ASR.

## Giấy phép và nguồn

Dữ liệu JointCapPunc được phát hành theo Apache-2.0. Bản giấy phép đi kèm nằm
tại [JOINTCAPPUNC_LICENSE.txt](JOINTCAPPUNC_LICENSE.txt).

PhoBERT sử dụng model `vinai/phobert-base-v2`; việc sử dụng và phân phối trọng
số tuân theo giấy phép do tác giả model công bố. Repository hiện chưa khai báo
một giấy phép riêng cho phần mã nguồn của dự án.
