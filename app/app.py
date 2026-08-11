from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples import EXAMPLES

st.set_page_config(
    page_title="Vietnamese Punctuation Restoration",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_INPUT_CHARS = 20000
INFERENCE_TIMEOUT_SECONDS = 300

def inject_design_system() -> None:
    """Apply a compact Streamlit design system for the NLP demo."""
    st.markdown(
        """
        <style>
        :root {
            --app-ink: #18202f;
            --app-muted: #657086;
            --app-border: #d9dfeb;
            --app-panel: #ffffff;
            --app-primary: #2457d6;
            --app-primary-dark: #173d9f;
            --app-accent: #0f8a7a;
        }

        .stApp {
            background: #f6f8fb;
            color: var(--app-ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.75rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--app-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] .stCaption {
            color: var(--app-muted);
        }

        .uupm-hero {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: var(--app-panel);
            padding: 1.2rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 22px rgba(24, 32, 47, 0.05);
        }

        .uupm-eyebrow {
            color: var(--app-accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .uupm-hero h1 {
            color: var(--app-ink);
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
            letter-spacing: 0;
            margin: 0 0 0.55rem 0;
        }

        .uupm-hero p {
            color: var(--app-muted);
            max-width: 760px;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .uupm-panel {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(24, 32, 47, 0.04);
        }

        .uupm-section-title {
            font-size: 1rem;
            font-weight: 800;
            color: var(--app-ink);
            margin: 0 0 0.25rem 0;
        }

        .uupm-helper {
            color: var(--app-muted);
            font-size: 0.92rem;
            line-height: 1.55;
            margin: 0 0 0.75rem 0;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--app-border);
            cursor: pointer;
            transition: background-color 180ms ease, border-color 180ms ease,
                color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--app-primary);
            color: var(--app-primary-dark);
            box-shadow: 0 8px 20px rgba(36, 87, 214, 0.12);
            transform: translateY(-1px);
        }

        .stButton > button:focus-visible,
        .stDownloadButton > button:focus-visible,
        textarea:focus {
            outline: 3px solid rgba(36, 87, 214, 0.22) !important;
            outline-offset: 2px;
        }

        .stTextArea textarea {
            border-radius: 8px;
            border-color: var(--app-border);
            line-height: 1.65;
            font-size: 1rem;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--app-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--app-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--app-ink);
            font-size: 1.35rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: #ffffff;
        }

        @media (max-width: 640px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .uupm-hero {
                padding: 1rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                transition: none !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel_start(title: str, helper: str | None = None) -> None:
    st.markdown('<section class="uupm-panel">', unsafe_allow_html=True)
    st.markdown(f'<p class="uupm-section-title">{title}</p>', unsafe_allow_html=True)
    if helper:
        st.markdown(f'<p class="uupm-helper">{helper}</p>', unsafe_allow_html=True)


def render_panel_end() -> None:
    st.markdown("</section>", unsafe_allow_html=True)






@st.cache_resource(show_spinner="Đang nạp mô hình… (chỉ chạy một lần)")
def load_service():
    """Load the locked winner once. Returns ``(service, error_message)``."""
    try:
        from src.inference.service import PunctuationService

        service = PunctuationService()
        _ = service.predictor
        return service, None
    except FileNotFoundError as exc:
        return None, (
            f"Không tìm thấy artifact cần thiết:\n\n`{exc}`\n\n"
            "Hãy chạy `notebooks/05_Model_Comparison_Selection.ipynb` để tạo "
            "`outputs/evaluation/model_selection.json`, và kiểm tra checkpoint của "
            "model thắng cuộc có tồn tại không."
        )
    except Exception as exc:
        return None, f"Không nạp được mô hình: `{type(exc).__name__}: {exc}`"


@st.cache_data(show_spinner=False)
def load_scores():
    """Validation/test scores for the sidebar, if those artifacts exist."""
    from src.utils.io import read_json

    out = {}
    for key, rel in (
        ("selection", "outputs/evaluation/model_selection.json"),
        ("test", "outputs/evaluation/final_test_results.json"),
    ):
        path = PROJECT_ROOT / rel
        if path.exists():
            try:
                out[key] = read_json(path)
            except Exception:
                pass
    return out


def _extract_json(stdout: str) -> dict:
    """Parse the JSON payload even if native libraries printed banners first."""
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Inference subprocess did not return JSON.")
    return json.loads(stdout[start : end + 1])


def _service_response_from_payload(data: dict) -> SimpleNamespace:
    return SimpleNamespace(
        ok=True,
        input_text=data.get("input_text", ""),
        restored_text=data.get("restored_text", ""),
        num_words=data.get("num_words", 0),
        num_sentences=data.get("num_sentences", 0),
        latency_ms=data.get("latency_ms", 0.0),
        experiment_id=data.get("experiment_id", ""),
        model_name=data.get("model_name", ""),
        error=None,
        error_type=None,
        details=data,
    )


def run_inference_subprocess(text: str) -> SimpleNamespace:
    """Run model inference out-of-process so Streamlit survives native crashes."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TQDM_DISABLE": "1",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        }
    )

    input_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        ) as f:
            f.write(text)
            input_path = f.name

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.inference.predict",
                "--file",
                input_path,
                "--json",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=INFERENCE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SimpleNamespace(
            ok=False,
            input_text=text,
            error=(
                "Inference subprocess timed out. Try a shorter input "
                "or run the CLI to check the checkpoint."
            ),
            error_type="inference_timeout",
            details={},
        )
    finally:
        if input_path:
            try:
                Path(input_path).unlink(missing_ok=True)
            except OSError:
                pass

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        if not stderr:
            stderr = f"Process exited with code {completed.returncode}."
        return SimpleNamespace(
            ok=False,
            input_text=text,
            error=(
                "Inference subprocess failed, but the Streamlit server is still alive. "
                f"Exit code: {completed.returncode}. {stderr[-1200:]}"
            ),
            error_type="inference_subprocess_failed",
            details={"returncode": completed.returncode},
        )

    try:
        return _service_response_from_payload(_extract_json(completed.stdout))
    except Exception as exc:
        return SimpleNamespace(
            ok=False,
            input_text=text,
            error=f"Could not parse inference JSON: {type(exc).__name__}: {exc}",
            error_type="invalid_inference_output",
            details={"stdout": completed.stdout[-1200:], "stderr": completed.stderr[-1200:]},
        )





def render_sidebar(service, scores) -> None:
    st.sidebar.header("Mô hình")

    selection = scores.get("selection", {})
    if service is None:
        st.sidebar.info("Mô hình sẽ được nạp khi bạn bấm khôi phục dấu câu.")
        if selection:
            st.sidebar.metric("Experiment thắng cuộc", selection.get("winner", "Chưa xác định"))
    else:
        info = service.model_info()
        st.sidebar.metric("Experiment thắng cuộc", info["experiment_id"])
        st.sidebar.write(f"**Model:** `{info['model_name']}`")
        st.sidebar.write(f"**Kiến trúc:** {info['model_type']}")
        if info.get("weight_mode"):
            st.sidebar.write(f"**Class weight:** `{info['weight_mode']}`")
        if info.get("model_revision"):
            st.sidebar.write(f"**Revision:** `{str(info['model_revision'])[:12]}…`")
        st.sidebar.write(f"**Thiết bị:** {info['device']} · fp16={info['fp16']}")
        st.sidebar.write(f"**Best epoch:** {info.get('best_epoch')}")

    st.sidebar.divider()
    st.sidebar.subheader("Cách chọn model")
    if selection:
        st.sidebar.write(
            f"Chọn trên **{selection.get('selection_split')}** bằng "
            f"**{selection.get('selection_metric')}**."
        )
        st.sidebar.write(
            f"Test **không** được dùng để chọn: "
            f"`test_was_used_for_selection = "
            f"{selection.get('test_was_used_for_selection')}`"
        )
        st.sidebar.metric(
            "Validation Punctuation Macro-F1",
            f"{selection.get('winner_validation_punctuation_macro_f1', 0):.4f}",
        )
    test = scores.get("test", {})
    if test:
        st.sidebar.metric(
            "Official test Punctuation Macro-F1",
            f"{test['metrics']['punctuation_macro_f1']:.4f}",
            help="Đo đúng một lần, sau khi winner đã được khoá bằng validation.",
        )

    st.sidebar.divider()
    st.sidebar.subheader("Giới hạn đã biết")
    st.sidebar.caption(
        "- Chỉ 4 nhãn: `O`, `,`, `.`, `?` - không có `!`, `:`, `;`, ngoặc kép.\n"
        "- Viết hoa là **rule-based** (đầu câu, sau `.`/`?`), không phải mô hình capitalization.\n"
        "- Dữ liệu huấn luyện là hội thoại tư vấn y tế; ngoài lĩnh vực này có thể kém chính xác hơn.\n"
        f"- Giới hạn đầu vào {MAX_INPUT_CHARS:,} ký tự; văn bản dài hơn bị **từ chối**."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Hướng mở rộng")
    st.sidebar.caption(
        "Bản này **chỉ xử lý văn bản**. Hướng mở rộng: "
        "`audio -> ASR -> transcript thô -> khôi phục dấu câu`."
    )




def set_input(text: str) -> None:
    """Prefill the input box (used by the example buttons)."""
    st.session_state["input_text"] = text


def clear_input() -> None:
    st.session_state["input_text"] = ""





def main() -> None:
    inject_design_system()
    st.title("Khôi phục dấu câu tiếng Việt")
    st.markdown(
        """
        <section class="uupm-hero">
            <div class="uupm-eyebrow">Demo suy luận NLP</div>
            <p>Khôi phục dấu câu cho văn bản tiếng Việt bằng mô hình được chọn trên tập validation. Giao diện tập trung vào nhập liệu, kết quả, thông tin thực nghiệm và các giới hạn đã biết.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    scores = load_scores()
    render_sidebar(None, scores)






    render_panel_start(
        "Ví dụ mẫu",
        "Chọn một mẫu để nạp nhanh văn bản đầu vào và kiểm tra mô hình trong từng ngữ cảnh.",
    )
    cols = st.columns(4)
    for i, example in enumerate(EXAMPLES):
        cols[i % 4].button(
            example["label"],
            key=f"example_{i}",
            help=example.get("note", ""),
            on_click=set_input,
            args=(example["text"],),
            use_container_width=True,
        )
    render_panel_end()


    render_panel_start(
        "Văn bản đầu vào",
        f"Tối đa {MAX_INPUT_CHARS:,} ký tự. Văn bản quá dài sẽ được từ chối rõ ràng, không bị cắt ngầm.",
    )
    text = st.text_area(
        "Nhập văn bản tiếng Việt không có dấu câu:",
        key="input_text",
        height=170,
        placeholder="hôm nay trời đẹp bạn có muốn đi dạo không",
        label_visibility="collapsed",
    )

    left, right = st.columns([1, 4])
    run = left.button(
        "Khôi phục dấu câu", key="restore_button", type="primary", use_container_width=True
    )
    right.button("Xóa", key="clear_button", on_click=clear_input)
    render_panel_end()

    if not run:
        return


    if not text or not text.strip():
        st.warning("Vui lòng nhập văn bản trước khi bấm nút.")
        return
    if len(text) > MAX_INPUT_CHARS:
        st.error(
            f"Văn bản dài {len(text):,} ký tự, vượt giới hạn {MAX_INPUT_CHARS:,}. "
            "Văn bản **không bị cắt bớt** - hãy chia nhỏ rồi thử lại."
        )
        return


    with st.spinner("Running punctuation restoration in a subprocess..."):
        response = run_inference_subprocess(text)

    if not response.ok:
        st.error(f"**{response.error_type}** — {response.error}")
        return

    render_panel_start("Kết quả", "Đầu vào và đầu ra được đặt cạnh nhau để dễ kiểm tra dấu phẩy, dấu chấm và dấu hỏi.")
    a, b = st.columns(2)
    with a:
        st.markdown("**Đầu vào (không dấu câu)**")
        st.text_area("input", value=response.input_text, height=190,
                     disabled=True, label_visibility="collapsed")
    with b:
        st.markdown("**Đã khôi phục dấu câu**")
        st.text_area("output", value=response.restored_text, height=190,
                     label_visibility="collapsed")

    st.success(response.restored_text)

    m1, m2, m3, m4, m5 = st.columns(5)
    details = response.details
    m1.metric("Số từ", f"{response.num_words:,}")
    m2.metric("Số câu", response.num_sentences)
    m3.metric("Dấu phẩy", details.get("num_commas", 0))
    m4.metric("Dấu hỏi", details.get("num_questions", 0))
    m5.metric("Độ trễ", f"{response.latency_ms:.0f} ms")
    render_panel_end()

    with st.expander("Chi tiết kỹ thuật"):
        st.write(
            f"- Experiment: **{response.experiment_id}** · model "
            f"`{response.model_name}` · thiết bị `{details.get('device')}`"
        )
        st.write(
            f"- Số cửa sổ suy luận: **{details.get('num_windows', 1)}** "
            "(văn bản dài được chia tại ranh giới từ, không bị cắt cụt)"
        )
        st.write(
            f"- Độ trễ **{response.latency_ms:.1f} ms** cho {response.num_words} từ "
            "— đo trên một lần chạy, không phải benchmark."
        )
        st.download_button(
            "Tải kết quả (.txt)",
            data=response.restored_text,
            file_name="restored.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
