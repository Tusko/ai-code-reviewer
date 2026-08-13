import re

from reviewer.diff_parser import FileDiff

LOCKFILE_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "composer.lock", "go.sum", "Gemfile.lock",
    "Pipfile.lock", "mix.lock",
})

ASSET_SUFFIXES = (
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".pdf", ".map",
    ".zip", ".gz", ".tar", ".jar", ".so", ".dylib", ".dll",
)

GENERATED_PATTERNS = (
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"\.pb\.go$"),
    re.compile(r"_pb2(_grpc)?\.py$"),
    re.compile(r"\.generated\.[^/]+$"),
    re.compile(r"\.g\.dart$"),
)

# Matched as whole path segments so 'src/distance.py' is not caught by 'dist'.
VENDOR_DIR_SEGMENTS = frozenset({
    "vendor", "dist", "build", "node_modules", "__snapshots__",
    ".venv", "venv", "site-packages", "third_party",
})


def is_reviewable(file_diff: FileDiff) -> tuple[bool, str]:
    """Returns (True, '') or (False, reason). Reason is shown in the summary note."""
    if file_diff.is_deleted:
        return False, "deleted"
    if file_diff.is_binary:
        return False, "binary"
    if not file_diff.hunks:
        return False, "no content change"

    path = file_diff.new_path
    segments = path.split("/")
    name = segments[-1]

    if name in LOCKFILE_NAMES or name.endswith(".lock"):
        return False, "lockfile"
    for pattern in GENERATED_PATTERNS:
        if pattern.search(path):
            return False, "generated"
    if any(segment in VENDOR_DIR_SEGMENTS for segment in segments[:-1]):
        return False, "vendored or build output"
    if path.endswith(ASSET_SUFFIXES):
        return False, "asset"

    if file_diff.total_lines == 0:
        return False, "no added lines"

    return True, ""
