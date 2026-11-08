import json
from pathlib import Path

from scripts.clinic_demo import main


def test_automated_cli_prints_redacted_transcripts_without_writing_log(capsys, monkeypatch):
    original_open = Path.open

    def reject_log_file(self, *args, **kwargs):
        assert self.name != "clinic_demo_automated.jsonl"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_log_file)
    assert main(["--automated-tests"]) == 0

    console_output = capsys.readouterr().out
    assert "6 cenários aprovados" in console_output
    for scenario_name in (
        "paciente_fluxos_1_a_4",
        "paciente_isolamento_da_agenda",
        "medico_fluxos_1_a_9",
        "medico_revisor_aprova_validacao",
        "cadastro_de_paciente_sintetico",
        "perfil_incompativel_com_credencial",
    ):
        assert scenario_name in console_output
    assert "Entradas do cenário:" in console_output
    assert "Saída do cenário:" in console_output
    assert "Resposta demonstrativa determinística" in console_output
    assert "Token de acesso: [REDACTED]" in console_output
    assert "333.666.999-50" not in console_output
    assert "autoteste@example.test" not in console_output
    assert "clinic_demo_automated.jsonl" not in console_output
    assert "nenhum arquivo de log é gravado" in console_output


def test_notebook_part_six_runs_automated_cli_without_input(monkeypatch, tmp_path):
    notebook_path = Path(__file__).resolve().parents[2] / "IADT 3 - Victor Luiz Domingues.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "clinic-demo-before-chat")
    source = "".join(cell["source"])
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr("scripts.clinic_demo.main", fake_main)
    namespace = {"PROJECT_DIR": tmp_path}
    exec(compile(source, str(notebook_path), "exec"), namespace)

    assert calls == [["--automated-tests"]]
    assert "input(" not in source
    assert "sqlite_demo" not in source
    assert namespace["clinic_demo_status"] == 0
