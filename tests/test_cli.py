from spam_detector import cli


def test_cli_exits_when_model_missing(monkeypatch, capsys) -> None:
    # Force MODEL_PATH to something missing
    monkeypatch.setattr(cli, "MODEL_PATH", cli.Path("model/does_not_exist.joblib"))
    cli.main()
    out = capsys.readouterr().out
    assert "Model file not found" in out or "Train first" in out

def test_cli_exits_on_empty_input(monkeypatch, capsys) -> None:
    from spam_detector import cli

    # mock model load
    monkeypatch.setattr(cli, "load_model", lambda _: object())

    # first input is empty string -> exit
    monkeypatch.setattr("builtins.input", lambda _: "")

    cli.main()
    out = capsys.readouterr().out.lower()

    assert "bye" in out


def test_cli_exits_on_eof(monkeypatch, capsys) -> None:
    from spam_detector import cli

    # mock model load
    monkeypatch.setattr(cli, "load_model", lambda _: object())

    def raise_eof(_: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)

    cli.main()
    out = capsys.readouterr().out.lower()

    assert "bye" in out
