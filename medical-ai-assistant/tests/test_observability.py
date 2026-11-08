import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import Settings
from src.utils.observability import build_callbacks, redact_payload


NOTEBOOK = Path(__file__).resolve().parents[2] / "IADT 3 - Victor Luiz Domingues.ipynb"


def test_redact_payload_is_recursive():
    out = redact_payload({"a": "CPF 111.444.777-35", "b": [{"c": "email x@y.com"}], "n": 3})
    assert "111.444" not in str(out) and "x@y.com" not in str(out) and out["n"] == 3


def test_langsmith_callback_disabled_by_default_does_not_change_global_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("MED_LANGSMITH_ENABLED", raising=False)
    assert build_callbacks(Settings()) == []
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_langsmith_requires_api_key(monkeypatch):
    monkeypatch.setenv("MED_LANGSMITH_ENABLED", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert build_callbacks(Settings()) == []


def test_langsmith_enabled_uses_tracer_with_injected_client(monkeypatch):
    monkeypatch.setenv("MED_LANGSMITH_ENABLED", "true")
    client = MagicMock()
    cbs = build_callbacks(Settings(), client=client)
    assert len(cbs) == 1 and cbs[0].client is client


def test_metadata_only_helper_omits_content():
    from src.utils.observability import _metadata_only

    assert "Maria" not in str(_metadata_only({"question": "Maria"}))


@pytest.mark.parametrize("send_content", [False, True])
def test_build_client_uses_default_client_configuration(send_content):
    with patch("langsmith.Client") as client_cls:
        from src.utils.observability import build_client

        assert build_client(send_content=send_content) is client_cls.return_value
        client_cls.assert_called_once_with()


@pytest.mark.parametrize(
    ("key", "callbacks", "error"),
    [
        ("synthetic-key", [object()], None),
        ("", [object()], "Configure o secret"),
        ("synthetic-key", [], "callback LangSmith não foi ativado"),
    ],
)
def test_notebook_langsmith_configuration(monkeypatch, capsys, key, callbacks, error):
    cell = next(cell for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
                if cell.get("id") == "uKgMYsetyvDZ")
    colab = ModuleType("google.colab")
    colab.userdata = SimpleNamespace(get=lambda name: key)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("MED_LANGSMITH_ENABLED", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("MED_LANGSMITH_SEND_CONTENT", raising=False)
    assistant = SimpleNamespace(callbacks=callbacks)
    namespace = {"Settings": Settings, "build_assistant": lambda settings: assistant}
    code = compile("".join(cell["source"]), str(NOTEBOOK), "exec")
    if error:
        with pytest.raises(RuntimeError, match=error):
            exec(code, namespace)
    else:
        exec(code, namespace)
        assert namespace["assistant"] is assistant
        assert "MED_LANGSMITH_ENABLED" not in os.environ
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["MED_LANGSMITH_SEND_CONTENT"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == key
        assert os.environ["LANGSMITH_PROJECT"] == "medical-ai-assistant"
        assert key not in capsys.readouterr().out


def test_notebook_requires_callback_enabled_separately(monkeypatch):
    cell = next(cell for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
                if cell.get("id") == "uKgMYsetyvDZ")
    colab = ModuleType("google.colab")
    colab.userdata = SimpleNamespace(get=lambda name: "synthetic-key")
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    for name in ("LANGSMITH_PROJECT", "LANGSMITH_API_KEY", "MED_LANGSMITH_ENABLED",
                 "LANGCHAIN_TRACING_V2", "MED_LANGSMITH_SEND_CONTENT"):
        monkeypatch.delenv(name, raising=False)
    namespace = {
        "Settings": Settings,
        "build_assistant": lambda settings: SimpleNamespace(callbacks=build_callbacks(settings, client=MagicMock())),
    }
    with pytest.raises(RuntimeError, match="callback LangSmith não foi ativado"):
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
