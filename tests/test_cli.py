from spam_detector import cli


def test_cli_exits_when_model_missing(monkeypatch, capsys) -> None:
    # Force MODEL_PATH to something missing
    monkeypatch.setattr(cli, "MODEL_PATH", cli.Path("model/does_not_exist.joblib"))
    cli.main()
    out = capsys.readouterr().out
    assert "Model file not found" in out or "Train first" in out
