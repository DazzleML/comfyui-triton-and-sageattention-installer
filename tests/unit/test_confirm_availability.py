"""Gap found while hunting for other "removes/overwrites without asking" defects
during the v0.8.13 checklist run (rollback-scope-and-locale-decoding).

`_confirm()` is defined exactly once in the file, on `ComfyUIInstaller`
(comfyui_triton_sageattention.py ~line 2532). `PlatformHandler` and its subclasses
(`WindowsHandler`, `LinuxHandler`, `MacOSHandler`) are a SEPARATE class hierarchy --
`ComfyUIInstaller` HAS-A handler instance, it does not inherit from it -- and none of
them define or inherit `_confirm`. Yet all three call `self._confirm(...)` directly,
in exactly the places meant to gate a destructive or disruptive action:

  WindowsHandler.install_build_tools()   line ~1255  -- force-reinstall Build Tools
  LinuxHandler.install_build_tools()     line ~1590  -- force-reinstall Build Tools
  LinuxHandler._setup_venv_environment() line ~1519  -- recreate an "invalid" venv
  MacOSHandler.install_build_tools()     line ~1959  -- force-reinstall Build Tools
  MacOSHandler._setup_venv_environment() line ~1893  -- recreate an "invalid" venv

Confirmed live and reproducible (not just a static read): calling
WindowsHandler.install_build_tools() with force=True, interactive=True, and Build
Tools already detected raises `AttributeError: 'WindowsHandler' object has no
attribute '_confirm'` -- on the PRIMARY, non-experimental platform this project
targets, not just the documented-experimental Linux/macOS stubs.

Two distinct consequences, both worth fixing but different in severity:

  1. INTERACTIVE mode: the confirm call itself raises before any destructive action
     runs, so this fails closed (no unwanted deletion) but hard-crashes instead of
     ever showing the prompt the code so clearly intended to show. The "ask before
     doing something destructive" gate has almost certainly never worked.

  2. NON-INTERACTIVE mode (Linux/macOS venv recreation only -- WindowsHandler's own
     _setup_venv_environment doesn't have an equivalent branch at all, see below):
     `self.interactive` is False, so the broken `_confirm` branch is skipped
     entirely and the code falls straight to `shutil.rmtree(self.venv_path)` with
     NO confirmation of any kind. An existing venv that this validation check
     considers "broken" (its `python --version` doesn't return "Python 3." --
     trivially true for a stale pyenv/asdf symlink whose target Python was
     upgraded or removed, while the venv's site-packages are still intact and
     valuable) gets permanently deleted, non-interactively, the very first time
     --install or --upgrade runs against it. This is exactly the class of defect
     v0.8.13 fixes elsewhere (Issue #36): removing something the user did not ask
     to have removed, this time via directory validation rather than a pip
     rollback.

WindowsHandler._setup_venv_environment() (~line 1175) has no rmtree-on-invalid
branch at all -- an "invalid" existing venv there falls through to `python -m venv`
being run again over the same directory, which is a third, milder inconsistency
(silent in-place mutation of a directory the code itself just flagged as broken,
neither deleted-with-consent nor left alone) but not a data-loss risk the way the
Linux/macOS branch is, so it is not asserted here.

All tests below encode the correct behavior. They FAILED when written and pass
as of the ConfirmMixin fix (v0.8.13); they are live regression tests now
-- they are the bug report, not a feature under test.
"""
import ast
import logging
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from comfyui_triton_sageattention import ComfyUIInstallerError, WindowsHandler, LinuxHandler, MacOSHandler

SRC = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"


def _classes_calling_self_confirm_without_defining_it():
    """Static structural scan: which classes call self._confirm(...) without
    _confirm being defined anywhere in their in-file base-class chain?"""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    class_defs = {n.name: n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.ClassDef)}

    def base_names(cls_node):
        return [b.id for b in cls_node.bases if isinstance(b, ast.Name)]

    def defines_method(cls_node, method_name):
        return any(isinstance(n, ast.FunctionDef) and n.name == method_name
                   for n in cls_node.body)

    def mro_defines(cls_name, method_name, seen=None):
        seen = seen or set()
        if cls_name in seen or cls_name not in class_defs:
            return False
        seen.add(cls_name)
        node = class_defs[cls_name]
        if defines_method(node, method_name):
            return True
        return any(mro_defines(b, method_name, seen) for b in base_names(node))

    offenders = []
    for cls_name, node in class_defs.items():
        calls_confirm = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_confirm" and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
            for method in node.body if isinstance(method, ast.FunctionDef)
            for n in ast.walk(method)
        )
        if calls_confirm and not mro_defines(cls_name, "_confirm"):
            offenders.append(cls_name)
    return sorted(offenders)


def test_every_class_calling_confirm_defines_or_inherits_it():
    offenders = _classes_calling_self_confirm_without_defining_it()
    assert not offenders, (
        f"these classes call self._confirm(...) without _confirm being defined "
        f"anywhere in their base-class chain: {offenders}"
    )


def test_windows_force_reinstall_build_tools_prompts_instead_of_crashing():
    """--install --force, interactive, on a machine that already has VS Build
    Tools: must show the confirm prompt, not crash before it appears."""
    h = object.__new__(WindowsHandler)
    h.logger = logging.getLogger("test")
    h.force = True
    h.interactive = True

    with patch.object(WindowsHandler, "_check_existing_build_tools", return_value=True), \
         patch("builtins.input", return_value="n"):
        try:
            h.install_build_tools()
        except AttributeError as e:
            pytest.fail(f"crashed before the confirm prompt could appear: {e}")


def test_linux_interactive_venv_recreate_prompts_instead_of_crashing(tmp_path):
    """An 'invalid' existing venv, interactive mode: must show the confirm prompt
    (and honor a decline), not crash before it appears."""
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("not a real interpreter")

    h = object.__new__(LinuxHandler)
    h.logger = logging.getLogger("test")
    h.base_path = tmp_path
    h.interactive = True
    h.may_create_env = True
    h.python_mode = "auto"
    h.python_explicit_path = None
    h.python_path = None
    h.venv_path = None
    h.environment_type = "unknown"
    h.run_command = Mock(side_effect=Exception("not executable"))

    with patch("builtins.input", return_value="n"):
        try:
            h._setup_venv_environment()
        except AttributeError as e:
            pytest.fail(f"crashed before the confirm prompt could appear: {e}")

    assert venv_dir.exists(), (
        "declining the recreate prompt must leave the existing venv in place"
    )


def test_non_interactive_venv_recreate_does_not_delete_without_consent(tmp_path):
    """Non-interactive mode must not silently destroy an existing venv's
    site-packages just because its interpreter fails a `--version` probe (e.g. a
    stale pyenv/asdf symlink) -- that is exactly the class of unasked-for removal
    v0.8.13 fixes elsewhere (Issue #36), via a different trigger."""
    venv_dir = tmp_path / "venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("not a real interpreter")
    marker = venv_dir / "lib" / "site-packages" / "precious_user_data.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("packages the user installed themselves")

    h = object.__new__(LinuxHandler)
    h.logger = logging.getLogger("test")
    h.base_path = tmp_path
    h.interactive = False  # non-interactive: no prompt, no crash -- straight to rmtree
    h.may_create_env = True
    h.python_mode = "auto"
    h.python_explicit_path = None
    h.python_path = None
    h.venv_path = None
    h.environment_type = "unknown"
    # Only the validation probe fails; venv creation is allowed to succeed so the
    # test exercises the disposal decision rather than dying on the rebuild.
    def only_probe_fails(cmd, *a, **k):
        if "--version" in [str(c) for c in cmd]:
            raise ComfyUIInstallerError("not executable")
        return Mock(returncode=0, stdout="", stderr="")
    h.run_command = Mock(side_effect=only_probe_fails)

    # The rebuild after disposal cannot succeed under a mock (no real interpreter
    # appears on disk), and it is not what this test is about. Let it fail; the
    # disposal decision has already been made by then.
    try:
        h._setup_venv_environment()
    except ComfyUIInstallerError:
        pass

    # The venv must be left exactly where it is. One failed `python --version`
    # probe is not enough evidence to delete, or even to rename, somebody's whole
    # environment -- the probe fails for transient reasons that say nothing about
    # whether the venv works.
    survivors = list(tmp_path.glob("venv/lib/site-packages/precious_user_data.txt"))
    assert survivors, (
        "non-interactive mode destroyed an existing venv (and the user data inside "
        "it) with no confirmation at all, just because the interpreter probe failed. "
        f"tmp_path now contains: {[p.name for p in tmp_path.iterdir()]}"
    )
    assert survivors[0].read_text() == "packages the user installed themselves"
