import importlib


def _reload(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import reviewer.config
    return importlib.reload(reviewer.config)


def test_defaults_match_16gb_tuning(monkeypatch):
    for key in ("OLLAMA_NUM_CTX", "OLLAMA_NUM_PREDICT", "OLLAMA_NUM_BATCH",
                "INCLUDE_FILE_CONTEXT", "CONTEXT_WINDOW", "MAX_FILES"):
        monkeypatch.delenv(key, raising=False)
    cfg = _reload(monkeypatch)
    assert cfg.OLLAMA_NUM_CTX == 8192
    assert cfg.OLLAMA_NUM_PREDICT == 320
    assert cfg.OLLAMA_NUM_BATCH == 512
    assert cfg.INCLUDE_FILE_CONTEXT is False
    assert cfg.CONTEXT_WINDOW == 15
    assert cfg.MAX_FILES == 40


def test_env_overrides_are_applied(monkeypatch):
    cfg = _reload(monkeypatch, OLLAMA_NUM_CTX="4096", INCLUDE_FILE_CONTEXT="true")
    assert cfg.OLLAMA_NUM_CTX == 4096
    assert cfg.INCLUDE_FILE_CONTEXT is True


def test_env_bool_accepts_common_truthy_spellings(monkeypatch):
    import reviewer.config as cfg
    monkeypatch.setenv("SOME_FLAG", "YES")
    assert cfg.env_bool("SOME_FLAG", False) is True
    monkeypatch.setenv("SOME_FLAG", "0")
    assert cfg.env_bool("SOME_FLAG", True) is False


def test_env_int_falls_back_on_garbage(monkeypatch):
    import reviewer.config as cfg
    monkeypatch.setenv("SOME_INT", "not-a-number")
    assert cfg.env_int("SOME_INT", 7) == 7


def test_memes_preserved():
    from reviewer.memes import meme_phrases
    assert len(meme_phrases) == 41
    assert "Nihuyasobi na oborot." in meme_phrases
