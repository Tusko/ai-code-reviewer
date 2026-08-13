from reviewer.diff_parser import parse_hunks, file_diff_from_change

MULTI_HUNK = """@@ -1,4 +1,5 @@
 def a():
-    return 1
+    return 2
+    # new

@@ -20,3 +21,4 @@ def b():
     x = 1
+    y = 2
     return x
"""

SINGLE_LINE_HUNK = """@@ -12 +12,3 @@
-old
+new_one
+new_two
+new_three
"""

WITH_FILE_HEADERS = """--- a/src/app.py
+++ b/src/app.py
@@ -5,2 +5,3 @@
 keep
+added
"""

NO_NEWLINE_AT_EOF = """@@ -1,2 +1,2 @@
 keep
-old
\\ No newline at end of file
+new
\\ No newline at end of file
"""

AT_IN_CONTEXT = """@@ -3,2 +3,3 @@ class Foo:  # @@ tricky @@
 keep
+added
"""


def test_parses_multiple_hunks():
    hunks = parse_hunks(MULTI_HUNK)
    assert len(hunks) == 2
    assert hunks[0].old_start == 1
    assert hunks[0].new_start == 1
    assert hunks[1].new_start == 21


def test_single_line_old_range_parses():
    # Regression for B4: git emits "@@ -12 +12,3 @@" with no comma on the old side.
    hunks = parse_hunks(SINGLE_LINE_HUNK)
    assert len(hunks) == 1
    assert hunks[0].old_start == 12
    assert hunks[0].new_start == 12


def test_added_lines_carry_correct_new_line_numbers():
    hunks = parse_hunks(MULTI_HUNK)
    assert hunks[0].added_lines() == [(2, "    return 2"), (3, "    # new")]
    assert hunks[1].added_lines() == [(22, "    y = 2")]


def test_removed_lines_do_not_advance_new_line_counter():
    hunks = parse_hunks(SINGLE_LINE_HUNK)
    assert hunks[0].added_lines() == [
        (12, "new_one"), (13, "new_two"), (14, "new_three"),
    ]


def test_file_headers_are_ignored():
    hunks = parse_hunks(WITH_FILE_HEADERS)
    assert len(hunks) == 1
    assert hunks[0].added_lines() == [(6, "added")]


def test_no_newline_marker_is_dropped():
    hunks = parse_hunks(NO_NEWLINE_AT_EOF)
    assert all(not line.startswith("\\") for line in hunks[0].lines)
    assert hunks[0].added_lines() == [(2, "new")]


def test_at_symbols_in_trailing_context_do_not_break_parsing():
    hunks = parse_hunks(AT_IN_CONTEXT)
    assert len(hunks) == 1
    assert hunks[0].new_start == 3
    assert hunks[0].added_lines() == [(4, "added")]


def test_first_added_line():
    hunks = parse_hunks(MULTI_HUNK)
    assert hunks[0].first_added_line() == 2


def test_first_added_line_is_none_when_only_deletions():
    hunks = parse_hunks("@@ -1,2 +1,1 @@\n keep\n-gone\n")
    assert hunks[0].first_added_line() is None


def test_file_diff_from_change_reads_gitlab_flags():
    fd = file_diff_from_change({
        "old_path": "src/app.py",
        "new_path": "src/app.py",
        "new_file": False,
        "deleted_file": False,
        "renamed_file": False,
        "diff": MULTI_HUNK,
    })
    assert fd.new_path == "src/app.py"
    assert fd.is_binary is False
    assert len(fd.hunks) == 2
    assert fd.total_lines == 3


def test_binary_diff_is_detected():
    fd = file_diff_from_change({
        "old_path": "logo.png",
        "new_path": "logo.png",
        "diff": "Binary files a/logo.png and b/logo.png differ\n",
    })
    assert fd.is_binary is True
    assert fd.hunks == ()


def test_empty_diff_yields_no_hunks():
    fd = file_diff_from_change({"old_path": "a", "new_path": "a", "diff": ""})
    assert fd.hunks == ()
    assert fd.total_lines == 0
