"""Build the evaluation PDF report from verified project artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import _bootstrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils.io import read_json

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
OUT = REPORT_DIR / "evaluation_report.pdf"

FONT_PAIRS = [
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
]


def rows(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, d=4):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def pct(x):
    return f"{float(x) * 100:.2f}%"


def p(text, style):
    return Paragraph(text, style)


def clean(text: str, limit: int | None = None) -> str:
    text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("…", "...").replace("—", "-")
    if limit and len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def table(data, widths=None, font_size=8.2, header=True):
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Arial"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8CDD6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF7")),
            ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10233F")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]
    t.setStyle(TableStyle(style))
    return t


def page_header(story, title, styles):
    story.append(p(title, styles["H1VN"]))
    story.append(Spacer(1, 0.12 * cm))


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(colors.HexColor("#5E6675"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "Khoi phuc dau cau tieng Viet")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Trang {doc.page}")
    canvas.restoreState()


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    font_pair = next(
        (
            (regular, bold)
            for regular, bold in FONT_PAIRS
            if regular.exists() and bold.exists()
        ),
        None,
    )
    if font_pair is None:
        raise FileNotFoundError("No Unicode TrueType font pair was found for PDF generation.")
    pdfmetrics.registerFont(TTFont("Arial", str(font_pair[0])))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_pair[1])))

    metrics = read_json(ROOT / "outputs" / "metrics.json")
    final = read_json(ROOT / "outputs" / "evaluation" / "final_report.json")
    stats = read_json(ROOT / "outputs" / "data" / "data_statistics.json")
    comparison = rows(ROOT / "outputs" / "model_comparison.csv")
    errors = rows(ROOT / "outputs" / "error_analysis.csv")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Cover", parent=styles["Title"], fontName="Arial-Bold", fontSize=24, leading=31, alignment=TA_CENTER, textColor=colors.HexColor("#10233F")))
    styles.add(ParagraphStyle("Sub", parent=styles["BodyText"], fontName="Arial", fontSize=12, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#465060")))
    styles.add(ParagraphStyle("H1VN", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=15.5, leading=20, textColor=colors.HexColor("#143860"), spaceAfter=8))
    styles.add(ParagraphStyle("H2VN", parent=styles["Heading2"], fontName="Arial-Bold", fontSize=11.2, leading=14, textColor=colors.HexColor("#10233F"), spaceBefore=4, spaceAfter=4))
    styles.add(ParagraphStyle("BodyVN", parent=styles["BodyText"], fontName="Arial", fontSize=9.4, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=5))
    styles.add(ParagraphStyle("SmallVN", parent=styles["BodyText"], fontName="Arial", fontSize=7.4, leading=9.2, alignment=TA_LEFT, spaceAfter=2))

    story = []

    story.append(Spacer(1, 1.1 * cm))
    story.append(p("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH KHÔI PHỤC DẤU CÂU TIẾNG VIỆT", styles["Cover"]))
    story.append(Spacer(1, 0.35 * cm))
    story.append(p("Đánh giá, phân tích lỗi, demo và tổng hợp", styles["Sub"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(table([["Hạng mục", "Kết quả chính"], ["Winner", f"{metrics['winner']} - {metrics['winner_model']}"], ["Official test Punctuation Macro-F1", fmt(metrics["punctuation_macro_f1"])], ["Accuracy tham khảo", fmt(metrics["accuracy_reference_only"])], ["Số lần chạy test", str(metrics["evaluation_runs_on_test"])], ["Trạng thái ASR", "Chưa có dữ liệu audio/transcript ASR trong repo"]], [5.2 * cm, 10.8 * cm], 9))
    story.append(Spacer(1, 0.35 * cm))
    story.append(p("Báo cáo này chỉ sử dụng số liệu đã được lưu trong thư mục outputs. Test chính thức được chạy một lần sau khi winner đã khóa bằng validation; do đó không dùng test để chọn mô hình.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "1. Phạm vi nhiệm vụ và tiêu chí hoàn thành", styles)
    story.append(p("Phần đánh giá chứng minh mô hình hoạt động đến đâu, giải thích lỗi, hoàn thiện hậu xử lý, demo và tổng hợp tài liệu bàn giao. Phạm vi kỹ thuật của dự án là gán nhãn dấu câu theo từng từ với bốn nhãn: O, COMMA, PERIOD và QUESTION.", styles["BodyVN"]))
    story.append(table([["Yêu cầu", "Trạng thái"], ["Precision/Recall/F1 từng dấu", "Đã có trong outputs/metrics.json"], ["Macro-F1 dấu câu", "Đã có, metric chính"], ["Confusion matrix", "Đã có CSV và PNG"], ["20-30 mẫu lỗi", "Đã có 30 mẫu trong outputs/error_analysis.csv"], ["Demo văn bản", "Đã có Streamlit và demo inference"], ["Demo âm thanh/ASR", "Chưa có dữ liệu đầu vào, ghi rõ không báo cáo số giả"], ["Báo cáo/slide", "File này và deck PPTX được tạo từ artifact"]], [6.2 * cm, 9.8 * cm], 8.6))
    story.append(p("Accuracy không được dùng làm chỉ số chính vì nhãn O chiếm hơn 91% token. Một mô hình đoán toàn O vẫn có accuracy cao nhưng F1 của các dấu câu bằng 0.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "2. Dữ liệu và phân bố nhãn", styles)
    s = stats["splits"]
    story.append(table([["Split", "Số mẫu", "Số token", "O", "COMMA", "PERIOD", "QUESTION"], ["Train", f"{s['train']['num_examples']:,}", f"{s['train']['num_tokens']:,}", pct(s['train']['label_ratios']['O']), pct(s['train']['label_ratios']['COMMA']), pct(s['train']['label_ratios']['PERIOD']), pct(s['train']['label_ratios']['QUESTION'])], ["Validation", f"{s['validation']['num_examples']:,}", f"{s['validation']['num_tokens']:,}", pct(s['validation']['label_ratios']['O']), pct(s['validation']['label_ratios']['COMMA']), pct(s['validation']['label_ratios']['PERIOD']), pct(s['validation']['label_ratios']['QUESTION'])], ["Test", f"{s['test']['num_examples']:,}", f"{s['test']['num_tokens']:,}", pct(s['test']['label_ratios']['O']), pct(s['test']['label_ratios']['COMMA']), pct(s['test']['label_ratios']['PERIOD']), pct(s['test']['label_ratios']['QUESTION'])]], [2.5 * cm, 2.4 * cm, 3 * cm, 2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], 8))
    story.append(p("Nguồn dữ liệu là JointCapPunc, giữ nguyên split chính thức. Phân bố nhãn rất lệch: QUESTION chỉ khoảng 0.56% token, trong khi O khoảng 91.1%. Điều này giải thích vì sao cần Macro-F1 trên ba dấu câu thay vì chỉ nhìn accuracy.", styles["BodyVN"]))
    story.append(p("Các artifact kèm theo lưu hash dữ liệu cho từng split để đảm bảo mọi thí nghiệm dùng cùng dữ liệu và không trộn test vào quá trình chọn mô hình.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "3. Các mô hình và thiết lập thí nghiệm", styles)
    exps = final["experiments"]
    story.append(table([["ID", "Mô hình", "Class weight", "Best epoch", "Val Punct-F1", "Val accuracy"], *[[e["experiment_id"], e["model"], e["weight_mode"], str(e["best_epoch"]), fmt(e["validation_punctuation_macro_f1"]), fmt(e["validation_accuracy"])] for e in exps]], [1.4 * cm, 5.5 * cm, 3 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm], 8))
    story.append(p("E1 là baseline BiLSTM. E2, E3 và E4 cùng dùng PhoBERT-base-v2 nhưng khác chiến lược trọng số lớp: không trọng số, inverse weight và sqrt-inverse weight. Mục tiêu là kiểm tra liệu tăng trọng số cho lớp hiếm có cải thiện F1 dấu câu hay không.", styles["BodyVN"]))
    story.append(p("Việc chọn winner dựa trên validation Punctuation Macro-F1. Loss dùng để tie-break được tính lại ở thang đo không trọng số để tránh so sánh loss gốc giữa các chế độ class weight khác nhau.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "4. Quy trình đánh giá", styles)
    story.append(p("Pipeline đánh giá tải checkpoint, chạy dự đoán trên split test, lưu nhãn thật và nhãn dự đoán, tính classification report, Macro-F1, confusion matrix và thời gian xử lý. Official test chỉ chạy sau khi winner E2 đã được khóa bằng validation.", styles["BodyVN"]))
    story.append(table([["Chỉ số", "Ý nghĩa"], ["Precision", "Tỷ lệ dự đoán dấu đúng trong các vị trí mô hình chọn"], ["Recall", "Tỷ lệ dấu thật được mô hình phát hiện"], ["F1-score", "Trung bình điều hòa giữa precision và recall"], ["Punctuation Macro-F1", "Trung bình F1 của COMMA, PERIOD, QUESTION"], ["Accuracy", "Chỉ số tham khảo vì bị lớp O chi phối"], ["Throughput", "Số mẫu/từ xử lý mỗi giây trên winner"]], [5.3 * cm, 10.7 * cm], 8.8))
    story.append(p(f"Thời gian đánh giá winner trên test: {metrics['timing']['evaluation_seconds']} giây, tương đương {fmt(metrics['timing']['examples_per_second'], 1)} mẫu/giây và {fmt(metrics['timing']['words_per_second'], 0)} từ/giây. Đây là timing cho full test của winner, không phải benchmark đầy đủ cho cả bốn checkpoint.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "5. Kết quả official test của winner", styles)
    per = metrics["per_punctuation_label"]
    story.append(table([["Nhãn", "Precision", "Recall", "F1", "Support"], *[[lab, fmt(per[lab]["precision"]), fmt(per[lab]["recall"]), fmt(per[lab]["f1"]), f"{int(per[lab]['support']):,}"] for lab in ["COMMA", "PERIOD", "QUESTION"]]], [3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm], 8.8))
    story.append(Spacer(1, 0.2 * cm))
    story.append(table([["Metric", "Giá trị"], ["Punctuation Macro-F1", fmt(metrics["punctuation_macro_f1"])], ["Macro-F1 cả 4 lớp", fmt(metrics["macro_f1_all_4_classes"])], ["Accuracy tham khảo", fmt(metrics["accuracy_reference_only"])], ["Số token test", f"{s['test']['num_tokens']:,}"]], [6 * cm, 5 * cm], 9))
    story.append(p("PERIOD là dấu dễ dự đoán nhất theo F1. QUESTION có recall cao nhất nhưng precision thấp hơn do một số câu kể/hỏi trong hội thoại y tế có tín hiệu ngữ nghĩa mơ hồ. COMMA khó nhất vì cách đặt dấu phẩy phụ thuộc văn phong và có nhiều phương án có thể chấp nhận.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "6. So sánh mô hình trên test", styles)
    story.append(table([["Mô hình", "F1 COMMA", "F1 PERIOD", "F1 QUESTION", "Punct-F1", "Accuracy", "Size MB"], *[[r["description"], fmt(r["f1_comma"]), fmt(r["f1_period"]), fmt(r["f1_question"]), fmt(r["punctuation_macro_f1"]), fmt(r["accuracy_reference_only"]), r["checkpoint_size_mb"]] for r in comparison]], [4.2 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.1 * cm, 2.1 * cm, 1.8 * cm], 7.8))
    story.append(p("PhoBERT không class weight (E2) đạt kết quả tốt nhất. So với BiLSTM, E2 tăng khoảng 0.1517 điểm Punctuation Macro-F1 trên test. Hai biến thể class weight không giúp trong thí nghiệm này; inverse weight làm giảm F1 mạnh hơn sqrt-inverse weight.", styles["BodyVN"]))
    story.append(p("Về tài nguyên, checkpoint BiLSTM nhỏ hơn nhiều (khoảng 16.9 MB) trong khi mỗi checkpoint PhoBERT khoảng 514.7 MB. Vì vậy BiLSTM nhẹ hơn để triển khai, còn PhoBERT cho chất lượng cao hơn rõ rệt.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "7. Confusion matrix và các nhầm lẫn chính", styles)
    cm_path = ROOT / "outputs" / "confusion_matrix.png"
    if cm_path.exists():
        story.append(Image(str(cm_path), width=10.8 * cm, height=8.8 * cm))
    top = final["error_analysis_summary"]["top_confusions"][:8]
    story.append(table([["Gold", "Pred", "Số lỗi"], *[[x["gold"], x["predicted"], f"{x['count']:,}"] for x in top]], [4 * cm, 4 * cm, 4 * cm], 8.3))
    story.append(p("Hai nhóm lớn nhất là COMMA -> O và O -> COMMA, cho thấy dấu phẩy vừa hay bị bỏ sót vừa hay bị thêm thừa. Đây cũng là lớp có F1 thấp nhất.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "8. Phân tích lỗi - mẫu 1 đến 15", styles)
    data = [["STT", "Gold", "Pred", "Loại lỗi", "Ngữ cảnh rút gọn"]]
    for r in errors[:15]:
        data.append([r["case_id"], r["gold_label"], r["predicted_label"], clean(r["error_type"], 34), p(clean(r["input_context"], 138), styles["SmallVN"])])
    story.append(table(data, [0.8 * cm, 1.5 * cm, 1.5 * cm, 3.4 * cm, 8.8 * cm], 6.8))
    story.append(PageBreak())

    page_header(story, "9. Phân tích lỗi - mẫu 16 đến 30", styles)
    data = [["STT", "Gold", "Pred", "Loại lỗi", "Ngữ cảnh rút gọn"]]
    for r in errors[15:30]:
        data.append([r["case_id"], r["gold_label"], r["predicted_label"], clean(r["error_type"], 34), p(clean(r["input_context"], 138), styles["SmallVN"])])
    story.append(table(data, [0.8 * cm, 1.5 * cm, 1.5 * cm, 3.4 * cm, 8.8 * cm], 6.8))
    story.append(PageBreak())

    page_header(story, "10. Hậu xử lý và demo", styles)
    story.append(p("Hậu xử lý chịu trách nhiệm biến chuỗi nhãn theo từ thành văn bản hoàn chỉnh: chèn dấu đúng vị trí, xóa khoảng trắng trước dấu, thêm khoảng trắng sau dấu, viết hoa đầu văn bản và sau dấu chấm/dấu hỏi, không tạo hai dấu câu liên tiếp, đồng thời bảo toàn URL, email, số thập phân và văn bản dài.", styles["BodyVN"]))
    story.append(table([["Thành phần", "File"], ["Inference CLI/API", "src/inference/predict.py, src/inference/predictor.py, src/inference/service.py"], ["Hậu xử lý", "src/inference/postprocess.py, src/inference/reconstruction.py"], ["Demo UI", "app/app.py"], ["Kiểm thử", "tests/test_reconstruction.py, test_predictor.py, test_app_smoke.py"]], [4.4 * cm, 11.6 * cm], 8.2))
    story.append(p("Ví dụ demo: input thiếu dấu 'bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài' được khôi phục thành hai câu, có dấu hỏi và dấu chấm. Demo Streamlit nạp winner từ model_selection.json, không hardcode checkpoint.", styles["BodyVN"]))
    story.append(PageBreak())

    page_header(story, "11. ASR, hạn chế và kết luận", styles)
    story.append(p("Đánh giá ASR chưa thực hiện vì repository hiện không có audio, transcript ASR hoặc nhãn tham chiếu. Báo cáo không đưa số liệu ASR giả; trạng thái đã được ghi riêng trong outputs/asr_evaluation_status.md cùng danh sách đầu vào cần có để hoàn thiện sau.", styles["BodyVN"]))
    story.append(table([["Kết luận", "Diễn giải"], ["Winner", "E2 PhoBERT không class weight là mô hình tốt nhất"], ["Kết quả chính", "Punctuation Macro-F1 = 0.7763 trên official test"], ["Class weight", "Không cải thiện trong thiết lập hiện tại"], ["Điểm yếu", "COMMA khó nhất, ASR/audio chưa có, dữ liệu chủ yếu miền y tế"], ["Bàn giao", "Code đánh giá, inference, hậu xử lý, outputs, demo, README, PDF và PPTX"]], [4.2 * cm, 11.8 * cm], 8.6))
    story.append(p("Hướng phát triển hợp lý là bổ sung tập ASR có align nhãn, chạy thêm nhiều seed, đánh giá ngoài miền y tế và huấn luyện thêm capitalization model thay vì chỉ viết hoa bằng luật.", styles["BodyVN"]))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=1.55 * cm, leftMargin=1.55 * cm, topMargin=1.45 * cm, bottomMargin=1.45 * cm, title="Bao cao NLP")
    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
