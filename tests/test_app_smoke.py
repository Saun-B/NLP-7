from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app" / "app.py"

pytestmark = pytest.mark.integration

def test_app_file_exists():
    assert APP_PATH.exists()

def test_app_imports_and_exposes_main():
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    sys.path.insert(0, str(PROJECT_ROOT))
    import importlib

    module = importlib.import_module("app.app") if (PROJECT_ROOT / "app" / "__init__.py").exists() else None
    if module is None:
        spec = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert callable(module.main)
    assert callable(module.load_service)

def test_winner_is_not_hardcoded_in_the_app():
    source = APP_PATH.read_text(encoding="utf-8")
    forbidden = re.findall(r'MODEL\s*=\s*["\']E[1-4]["\']', source)
    assert not forbidden, f"winner hardcoded in app.py: {forbidden}"
    assert not re.search(r'winner\s*=\s*["\']E[1-4]["\']', source, re.IGNORECASE)

    assert "PunctuationService" in source
    assert 'selection.get("winner",' in source
    assert "winner_experiment_id" not in source

def test_model_is_cached():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "@st.cache_resource" in source, "the model must be cached across reruns"

def test_examples_are_usable():
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    from examples import EXAMPLES, EXAMPLE_TEXTS

    assert len(EXAMPLES) >= 5
    for e in EXAMPLES:
        assert e["label"] and e["text"]
        assert len(e["text"].split()) >= 5

        assert not any(ch in e["text"] for ch in "?!"), e["label"]

        assert "restored" not in e and "output" not in e
    assert len(set(EXAMPLE_TEXTS)) == len(EXAMPLE_TEXTS), "duplicate examples"

def test_examples_include_a_question_and_a_multi_clause_case():
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    from examples import EXAMPLE_TEXTS

    assert any(t.rstrip().endswith("không") or " không " in t for t in EXAMPLE_TEXTS)
    assert any(len(t.split()) > 30 for t in EXAMPLE_TEXTS), "need a long example"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

@pytest.mark.slow
def test_app_renders_headlessly_and_restores_text():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.evaluation.selection import MODEL_SELECTION_PATH

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present - run notebook 05")

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()

    assert not at.exception, f"app raised on first render: {at.exception}"
    assert any(("Punctuation Restoration" in t.value) or ("Khôi phục dấu câu" in t.value) for t in at.title), "title missing"

    assert at.sidebar.metric, "sidebar model card did not render"

    at.text_area(key="input_text").set_value("hôm nay trời đẹp bạn có muốn đi dạo không")
    at.button(key="restore_button").click().run()

    assert not at.exception, f"app raised while restoring: {at.exception}"

    rendered = " ".join(
        [e.value for e in at.success] + [ta.value for ta in at.text_area]
    )
    assert any(ch in rendered for ch in ".?,"), (
        f"no punctuation appeared in the rendered output: {rendered[:300]}"
    )
    assert "trời" in rendered, "input words missing from the output"

@pytest.mark.slow
def test_app_shows_a_warning_for_empty_input():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.evaluation.selection import MODEL_SELECTION_PATH

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present")

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    at.text_area(key="input_text").set_value("   ")
    at.button(key="restore_button").click().run()

    assert not at.exception
    assert at.warning, "empty input should produce a visible warning, not a crash"

@pytest.mark.slow
def test_app_example_buttons_prefill_the_input():
    """Clicking an example must fill the box without raising a widget-state error."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.evaluation.selection import MODEL_SELECTION_PATH

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present")

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    at.button(key="example_0").click().run()

    assert not at.exception, f"example button raised: {at.exception}"
    assert at.text_area(key="input_text").value.strip(), "example did not prefill the input"

@pytest.mark.slow
def test_app_clear_button_empties_the_input():
    """The clear button must not raise 'cannot be modified after instantiation'."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.evaluation.selection import MODEL_SELECTION_PATH

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present")

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    at.text_area(key="input_text").set_value("hôm nay trời đẹp")
    at.button(key="clear_button").click().run()

    assert not at.exception, f"clear button raised: {at.exception}"
    assert at.text_area(key="input_text").value == ""

@pytest.mark.slow
def test_streamlit_server_starts():
    """Launch the real app and confirm it serves HTTP before shutting it down."""
    pytest.importorskip("streamlit")
    from src.evaluation.selection import MODEL_SELECTION_PATH

    if not MODEL_SELECTION_PATH.exists():
        pytest.skip("model_selection.json not present - run notebook 05")

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(APP_PATH),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 120
        serving = False
        while time.time() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"streamlit exited early:\n{output[:2000]}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    serving = True
                    break
            except OSError:
                time.sleep(1)
        assert serving, "streamlit did not start listening within 120s"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
