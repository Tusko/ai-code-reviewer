import pytest

from reviewer.diff_parser import FileDiff, Hunk
from reviewer.filters import is_reviewable

HUNK = Hunk(old_start=1, new_start=1, lines=(" keep", "+added"))


def make(path, **kwargs):
    return FileDiff(
        old_path=kwargs.pop("old_path", path),
        new_path=path,
        is_new=kwargs.pop("is_new", False),
        is_deleted=kwargs.pop("is_deleted", False),
        is_renamed=kwargs.pop("is_renamed", False),
        is_binary=kwargs.pop("is_binary", False),
        hunks=kwargs.pop("hunks", (HUNK,)),
    )


def test_normal_source_file_is_reviewable():
    assert is_reviewable(make("src/app.py")) == (True, "")


def test_deleted_file_is_skipped():
    ok, reason = is_reviewable(make("src/app.py", is_deleted=True))
    assert ok is False
    assert reason == "deleted"


def test_binary_file_is_skipped():
    ok, reason = is_reviewable(make("logo.png", is_binary=True, hunks=()))
    assert ok is False
    assert reason == "binary"


def test_rename_without_content_change_is_skipped():
    ok, reason = is_reviewable(make("b.py", old_path="a.py", is_renamed=True, hunks=()))
    assert ok is False
    assert reason == "no content change"


@pytest.mark.parametrize("path", [
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    "Gemfile.lock",
    "composer.lock",
])
def test_lockfiles_are_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "lockfile"


@pytest.mark.parametrize("path", [
    "static/app.min.js",
    "static/app.min.css",
    "api/service.pb.go",
    "api/service_pb2.py",
    "src/schema.generated.ts",
])
def test_generated_files_are_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "generated"


@pytest.mark.parametrize("path", [
    "vendor/lib/x.go",
    "dist/bundle.js",
    "build/out.js",
    "node_modules/pkg/index.js",
    "tests/__snapshots__/App.test.js.snap",
    ".venv/lib/site.py",
])
def test_vendored_and_build_output_is_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "vendored or build output"


@pytest.mark.parametrize("path", [
    "assets/logo.svg",
    "assets/photo.jpg",
    "assets/icon.ico",
    "fonts/x.woff2",
    "docs/manual.pdf",
    "static/app.js.map",
])
def test_asset_files_are_skipped(path):
    ok, reason = is_reviewable(make(path))
    assert ok is False
    assert reason == "asset"


@pytest.mark.parametrize("path", [
    "db/migrations/001_add_users.sql",
    "src/vendored_client.py",
    "src/distance.py",
    "src/buildings.py",
])
def test_lookalike_paths_are_not_skipped(path):
    # 'migrations' stays reviewable: raw SQL is exactly what we want reviewed.
    # Substring matches like 'distance' must not trip the 'dist/' rule.
    assert is_reviewable(make(path)) == (True, "")


def test_deletion_only_file_is_skipped():
    # I4 regression: a file whose change is purely removals has zero '+'
    # lines once removed lines are stripped from the prompt, costing a full
    # inference for nothing — and select_files sorts it first since
    # total_lines (added-only) is 0.
    deletion_only = Hunk(old_start=1, new_start=1, lines=(" keep", "-removed"))
    ok, reason = is_reviewable(make("src/app.py", hunks=(deletion_only,)))
    assert ok is False
    assert reason == "no added lines"
