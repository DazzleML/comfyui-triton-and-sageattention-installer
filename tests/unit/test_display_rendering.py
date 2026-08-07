"""Console rendering tests -- the box must not come apart.

The `--show-installed` table used fixed 15/28/14 column widths with a hardcoded
67-character rule above and below it. SageAttention's local version part is
routinely longer than 28 characters:

    2.2.0+cu130torch2.9.0andhigher.post4     (36)
    2.2.0+cu130torch2.10.0andhigher.post6    (37)

Python's `:<28` pads but never truncates, so those rows simply ran past the
rules and the right-hand border stuck out. Reported from a real run.

These tests assert the property that actually matters -- every line of the table
is the same width -- rather than any particular width, so adding a column or a
longer version string cannot silently break alignment again.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from comfyui_triton_sageattention import ComfyUIInstaller


HEADERS = ("Component", "Version", "Status")
MIN_WIDTHS = (15, 28, 14)


def render(rows):
    """Mirror of the table rendering in show_installed()."""
    widths = [
        max([floor, len(head)] + [len(row[i]) for row in rows])
        for i, (head, floor) in enumerate(zip(HEADERS, MIN_WIDTHS))
    ]
    table_width = sum(widths) + 10
    out = [
        "=" * table_width,
        f"| {HEADERS[0]:<{widths[0]}} | {HEADERS[1]:<{widths[1]}} | {HEADERS[2]:<{widths[2]}} |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out += [
        f"| {c:<{widths[0]}} | {v:<{widths[1]}} | {s:<{widths[2]}} |"
        for c, v, s in rows
    ]
    out.append("=" * table_width)
    return out


EMPTY_ENV = [
    ("SageAttention", "-", "Not installed"),
    ("Triton", "-", "Not installed"),
    ("PyTorch", "-", "Not installed"),
    ("CUDA", "12.8", "Detected"),
    ("Python", "3.12.0", "Active"),
]

# The exact rows that broke it, from a real ComfyUI install.
LONG_SA_ENV = [
    ("SageAttention", "2.2.0+cu130torch2.9.0andhigher.post4", "Installed"),
    ("Triton", "3.6.0.post26", "Installed"),
    ("PyTorch", "2.10.0", "Installed"),
    ("CUDA", "13.0", "Detected"),
    ("Python", "3.12.0", "Active"),
]

ABSURD_ENV = [
    ("SageAttention", "3.0.0+cu999torch9.9.9andhigher.post999+localtag.extra", "Installed"),
    ("Triton", "3.7.1.post27", "Installed"),
    ("PyTorch", "2.13.0", "Installed"),
    ("CUDA", "13.2", "Detected"),
    ("Python", "3.13.11", "Active"),
]


@pytest.mark.parametrize("name,rows", [
    ("empty", EMPTY_ENV),
    ("long_sageattention_version", LONG_SA_ENV),
    ("absurdly_long_version", ABSURD_ENV),
])
def test_table_lines_are_all_the_same_width(name, rows):
    """Rules, header, separator and every data row must line up exactly."""
    lines = render(rows)
    widths = {len(line) for line in lines}
    assert len(widths) == 1, (
        f"[{name}] table lines have {len(widths)} different widths {sorted(widths)}; "
        f"the border will not line up:\n" + "\n".join(lines)
    )


@pytest.mark.parametrize("name,rows", [
    ("empty", EMPTY_ENV),
    ("long_sageattention_version", LONG_SA_ENV),
    ("absurdly_long_version", ABSURD_ENV),
])
def test_no_value_is_truncated(name, rows):
    """Widening is the chosen fix -- values must never be cut off to fit."""
    body = "\n".join(render(rows))
    for component, version, status in rows:
        assert version in body, f"[{name}] version {version!r} was truncated"
        assert component in body, f"[{name}] component {component!r} was truncated"
        assert status in body, f"[{name}] status {status!r} was truncated"


def test_short_content_keeps_the_original_width():
    """An environment with nothing installed renders exactly as it always did.

    The fix keeps the historical 15/28/14 widths as minimums so the common case
    is byte-identical to previous releases; the box grows only when it must.
    """
    lines = render(EMPTY_ENV)
    assert len(lines[0]) == 67, (
        f"short-content table is {len(lines[0])} wide, was 67 before the change"
    )


def test_rendering_matches_the_installer():
    """Guard against this mirror drifting from show_installed()'s real code."""
    import inspect
    src = inspect.getsource(ComfyUIInstaller.show_installed)
    assert "min_widths = (15, 28, 14)" in src, (
        "show_installed() no longer uses the minimum widths this test mirrors"
    )
    assert "sum(widths) + 10" in src, (
        "show_installed() no longer computes table_width the way this test does"
    )
