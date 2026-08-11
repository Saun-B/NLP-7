from __future__ import annotations

from typing import Dict, List

EXAMPLES: List[Dict[str, str]] = [
    {
        "label": "Câu kể + câu hỏi",
        "note": "Cần một PERIOD rồi một QUESTION.",
        "text": "hôm nay trời đẹp bạn có muốn đi dạo không",
    },
    {
        "label": "Bài tập nhóm",
        "note": "Câu hỏi kết thúc bằng 'chưa', sau đó là một câu kể.",
        "text": "bạn đã hoàn thành bài tập chưa ngày mai chúng ta nộp bài",
    },
    {
        "label": "Giới thiệu",
        "note": "Một câu dài, cần dấu phẩy ngăn mệnh đề.",
        "text": "xin chào tôi đang thử mô hình khôi phục dấu câu tiếng việt "
                "mô hình này được huấn luyện trên bộ dữ liệu jointcappunc",
    },
    {
        "label": "Hội thoại y tế (đúng domain)",
        "note": "Đúng lĩnh vực dữ liệu huấn luyện — kỳ vọng kết quả tốt nhất.",
        "text": "chào bác sĩ em bị đau bụng mấy hôm nay rồi em có nên đi khám không "
                "em cảm ơn bác sĩ nhiều",
    },
    {
        "label": "Nhiều mệnh đề",
        "note": "Kiểm tra dấu phẩy trước các liên từ 'nhưng', 'nếu'.",
        "text": "tôi muốn đi chơi nhưng trời đang mưa rất to nếu chiều nay tạnh "
                "thì chúng ta sẽ đi công viên",
    },
    {
        "label": "Có số và đơn vị",
        "note": "Tokenizer phải giữ nguyên '38.5' thành một từ.",
        "text": "em bị sốt 38.5 độ từ tối qua uống thuốc hạ sốt rồi mà chưa đỡ "
                "em phải làm sao ạ",
    },
    {
        "label": "Ngoài domain (tin tức)",
        "note": "Khác lĩnh vực huấn luyện — dùng để thấy giới hạn của mô hình.",
        "text": "giá xăng trong nước tiếp tục tăng mạnh trong kỳ điều hành hôm nay "
                "nhiều doanh nghiệp vận tải cho biết họ sẽ phải điều chỉnh giá cước",
    },
    {
        "label": "Văn bản dài (nhiều câu)",
        "note": "Kiểm tra xử lý văn bản dài và tách nhiều câu.",
        "text": "em chào bác sĩ ạ em năm nay 25 tuổi gần đây em hay bị mất ngủ "
                "khó vào giấc và hay tỉnh giấc lúc nửa đêm ban ngày em thấy mệt "
                "mỏi khó tập trung làm việc em có nên đi khám chuyên khoa thần "
                "kinh không hay chỉ cần điều chỉnh sinh hoạt thôi ạ em xin cảm ơn "
                "bác sĩ",
    },
]


EXAMPLE_TEXTS: List[str] = [e["text"] for e in EXAMPLES]


def get_example(label: str) -> str:
    for e in EXAMPLES:
        if e["label"] == label:
            return e["text"]
    raise KeyError(f"Unknown example {label!r}")


LABELS: List[str] = [e["label"] for e in EXAMPLES]
