import re
import importlib

import pytest


@pytest.fixture(autouse=True)
def _restore_config_module_after_test():
    """Undoes the module-level mutation importlib.reload() leaves behind.

    monkeypatch.setenv is undone at test teardown, but the already-reloaded
    reviewer.config module is not automatically reloaded back — so without
    this, every test file that runs after this one would see whatever env
    override the last test here happened to apply (e.g. OLLAMA_NUM_CTX=4096).
    This fixture is set up before each test's own monkeypatch fixture (both
    are function-scoped, and this one is autouse) and so is torn down after
    it, once monkeypatch has already restored the real environment — the
    reload here then picks up the clean environment.
    """
    yield
    import reviewer.config
    importlib.reload(reviewer.config)


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
    assert len(meme_phrases) > 40
    assert "Nihuyasobi na oborot." in meme_phrases
    assert len(set(meme_phrases)) == len(meme_phrases)


def test_no_meme_phrase_was_glued_by_a_missing_comma():
    """A dropped comma silently concatenates two adjacent literals."""
    from reviewer.memes import meme_phrases
    for phrase in meme_phrases:
        assert not re.search(r"[.!?][\u0410-\u042f\u0406\u0407\u0404A-Z]", phrase), phrase


def test_openrouter_defaults(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_BASE_URL",
                "OPENROUTER_MAX_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    cfg = _reload(monkeypatch)
    assert cfg.OPENROUTER_API_KEY is None
    assert cfg.OPENROUTER_MODEL == "google/gemini-2.5-flash-lite"
    assert cfg.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert cfg.OPENROUTER_MAX_TOKENS == 512


def test_openrouter_env_overrides(monkeypatch):
    cfg = _reload(
        monkeypatch,
        OPENROUTER_API_KEY="sk-or-test",
        OPENROUTER_MODEL="google/gemma-4-31b-it",
        OPENROUTER_MAX_TOKENS="256",
    )
    assert cfg.OPENROUTER_API_KEY == "sk-or-test"
    assert cfg.OPENROUTER_MODEL == "google/gemma-4-31b-it"
    assert cfg.OPENROUTER_MAX_TOKENS == 256


def test_snark_defaults_on():
    import reviewer.config as cfg
    assert cfg.SNARK is True


def test_snark_returns_a_meme_phrase():
    from reviewer.memes import meme_phrases, snark
    assert snark() in meme_phrases


def test_env_list_splits_and_trims(monkeypatch):
    import reviewer.config as cfg
    monkeypatch.setenv("OR_MODELS", " a/b , c/d:free ,, ")
    assert cfg.env_list("OR_MODELS", []) == ["a/b", "c/d:free"]


def test_env_list_falls_back_when_blank_or_unset(monkeypatch):
    import reviewer.config as cfg
    monkeypatch.setenv("OR_MODELS", "   ,  ")
    assert cfg.env_list("OR_MODELS", ["x"]) == ["x"]
    monkeypatch.delenv("OR_MODELS", raising=False)
    assert cfg.env_list("OR_MODELS", ["x"]) == ["x"]


def test_fallback_models_default_empty_and_parse(monkeypatch):
    monkeypatch.delenv("OPENROUTER_FALLBACK_MODELS", raising=False)
    assert _reload(monkeypatch).OPENROUTER_FALLBACK_MODELS == [
        "google/gemma-4-26b-a4b-it",
    ], "the default fallback is a different family, so an outage reroutes"
    # A model distinct from OPENROUTER_MODEL: a fallback equal to the primary
    # is now dropped, since rerouting from a model to itself buys nothing.
    cfg = _reload(monkeypatch, OPENROUTER_FALLBACK_MODELS="anthropic/claude-sonnet-5")
    assert cfg.OPENROUTER_FALLBACK_MODELS == ["anthropic/claude-sonnet-5"]


def test_a_free_model_is_refused_however_it_is_configured(monkeypatch):
    """"тільки нормальні" has to hold for anything an operator can type, not
    just for the default. Free variants share one saturated upstream pool and
    429 constantly, which trips the voice breaker and leaves half an MR dry."""
    cfg = _reload(
        monkeypatch,
        OPENROUTER_MODEL="google/gemma-4-26b-a4b-it:free",
        OPENROUTER_REVIEW_MODEL="poolside/laguna-s-2.1:free",
        OPENROUTER_FALLBACK_MODELS="a/b:free, c/d",
        OPENROUTER_REVIEW_FALLBACK_MODELS="e/f:free",
    )
    assert cfg.OPENROUTER_MODEL == "google/gemma-4-26b-a4b-it"
    assert cfg.OPENROUTER_REVIEW_MODEL == "poolside/laguna-s-2.1"
    assert cfg.OPENROUTER_FALLBACK_MODELS == ["a/b", "c/d"]
    assert cfg.OPENROUTER_REVIEW_FALLBACK_MODELS == ["e/f"]


def test_a_fallback_that_collapses_onto_the_primary_is_dropped(monkeypatch):
    """The old advice was to name the paid slug as the fallback for a :free
    primary. Once :free is stripped both become the same model, and asking
    OpenRouter to reroute from a model to itself buys nothing."""
    cfg = _reload(
        monkeypatch,
        OPENROUTER_MODEL="google/gemma-4-26b-a4b-it:free",
        OPENROUTER_FALLBACK_MODELS="google/gemma-4-26b-a4b-it",
    )
    assert cfg.OPENROUTER_FALLBACK_MODELS == []


def test_the_voice_default_is_never_a_free_or_batch_endpoint(monkeypatch):
    """Batch endpoints are async and the voice has a 20s deadline; free
    variants share a saturated pool and trip the breaker."""
    cfg = _reload(monkeypatch)
    for model in [cfg.OPENROUTER_MODEL, *cfg.OPENROUTER_FALLBACK_MODELS]:
        assert not model.endswith(":free"), model
        assert not model.endswith(":batch"), model
