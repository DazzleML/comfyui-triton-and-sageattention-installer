"""Confirmation-prompt tests -- the guard on every destructive action.

The default-yes prompts compared `response.lower() == 'n'`, so only the single
character "n" cancelled. Typing "no" -- the most natural negative answer there
is -- did not match, and the destructive change went ahead and printed
"Success!". Found by driving the real downgrade-consent prompt with piped stdin,
not by any mocked test.

The property under test: nothing but an unambiguous affirmative may be read as
consent, and the absence of an answer is never consent.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from comfyui_triton_sageattention import ComfyUIInstaller


@pytest.fixture
def installer():
    return object.__new__(ComfyUIInstaller)


def answer(installer, text, default_yes):
    """Run _confirm with `text` on stdin."""
    original = sys.stdin
    try:
        sys.stdin = io.StringIO(text)
        return installer._confirm("Continue?", default_yes=default_yes)
    finally:
        sys.stdin = original


NEGATIVES = ["n\n", "N\n", "no\n", "NO\n", "No\n", "  no  \n", "nope\n", "maybe\n"]
AFFIRMATIVES = ["y\n", "Y\n", "yes\n", "YES\n", "Yes\n", "  yes  \n"]


@pytest.mark.parametrize("reply", NEGATIVES)
def test_negative_answers_never_consent(installer, reply):
    """This is the bug: "no" must cancel a default-YES destructive prompt."""
    assert answer(installer, reply, default_yes=True) is False, (
        f"{reply!r} was read as consent to a destructive change"
    )
    assert answer(installer, reply, default_yes=False) is False


@pytest.mark.parametrize("reply", AFFIRMATIVES)
def test_affirmative_answers_consent(installer, reply):
    """"yes" must work, not just "y" -- the default-no prompts rejected it."""
    assert answer(installer, reply, default_yes=False) is True, (
        f"{reply!r} was not accepted as agreement"
    )
    assert answer(installer, reply, default_yes=True) is True


def test_empty_answer_takes_the_default(installer):
    assert answer(installer, "\n", default_yes=True) is True
    assert answer(installer, "\n", default_yes=False) is False


def test_unrecognized_answer_is_not_consent(installer):
    """Ambiguity must fail closed, even on a default-yes prompt."""
    assert answer(installer, "purple\n", default_yes=True) is False


def test_eof_cancels_and_does_not_raise(installer):
    """No controlling terminal (cron, CI, exhausted pipe) must not proceed.

    Previously this raised an uncaught EOFError traceback. It failed safe, but
    ungracefully -- and a destructive step must never proceed merely because
    nobody was present to object.
    """
    for default_yes in (True, False):
        assert answer(installer, "", default_yes=default_yes) is False


def test_no_raw_yes_no_comparisons_remain_in_the_source():
    """Every prompt must route through _confirm, not hand-rolled comparisons.

    The bug existed in 2 of 15 prompts; the other 13 were safe only by luck of
    polarity. A new prompt written in the old style would reintroduce it, so
    this asserts structurally that the pattern is gone.

    Parsed with `ast` rather than grepped, so prose in docstrings describing the
    old bug (there is some, deliberately) cannot trip it.
    """
    import ast

    src = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    def is_lower_call(node):
        return (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "lower")

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(is_lower_call(o) for o in operands):
            continue
        literals = [o.value for o in operands
                    if isinstance(o, ast.Constant) and isinstance(o.value, str)]
        if any(lit.strip().lower() in ("y", "n", "yes", "no") for lit in literals):
            offenders.append(f"line {node.lineno}: compares .lower() against {literals}")

    assert not offenders, (
        "hand-rolled yes/no comparisons found; route them through self._confirm():\n  "
        + "\n  ".join(offenders)
    )


def test_every_prompt_uses_the_helper():
    """No `input(...)` should ask a yes/no question outside _confirm."""
    import ast

    src = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "input" and node.args):
            continue
        arg = node.args[0]
        text = arg.value if isinstance(arg, ast.Constant) else ast.dump(arg)
        if "(y/N)" in text or "(Y/n)" in text:
            offenders.append(f"line {node.lineno}: {text!r}")

    assert not offenders, (
        "yes/no prompts bypassing self._confirm():\n  " + "\n  ".join(offenders)
    )
