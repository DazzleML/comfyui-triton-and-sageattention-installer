"""Live check: does the interactive downgrade-consent prompt actually gate the change?

Checklist reference: tests/checklists/v0.8.12__Epic__plan-execute-integrity.md, Section 2.2.
That section is marked "human only" because automated tests mock subprocesses and can't
drive a real input() prompt. This script drives the REAL prompt with real piped stdin
against a REAL environment that has SageAttention 2.x installed, and empirically checks:

  1. Does exact "n" cancel and leave the environment untouched? (the documented behavior)
  2. Does exact "N" (uppercase) also cancel?
  3. Does the natural human response "no" ALSO cancel -- or does it slip through?
  4. Does closing stdin (EOF, e.g. no controlling terminal) cancel safely, or crash /
     silently proceed?

The prompt gate lives in ComfyUIInstaller.install() and is a single exact-match
comparison: `if response.lower() == 'n':`. That is a suspicious pattern -- it does not
match "no", trailing whitespace, or anything but the single character 'n'/'N'. This
script proves empirically whether that suspicion is a real bypass.

DESTRUCTIVE: if a bypass is real, this script's test 3 (and test 4, if EOF also
proceeds) will genuinely downgrade the target environment's SageAttention to 1.0.6.
Point --base-path at a disposable environment. Tests run in order n -> N -> no -> EOF
and stop early once the environment has actually changed (no point burning further
scenarios once the SA version is gone).

Usage:
  python tests/one-offs/test_interactive_consent_live.py --base-path "<ComfyUI dir with SA2.x>"
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
INSTALLER = REPO / "comfyui_triton_sageattention.py"


def venv_python(base_path: Path) -> Path:
    return base_path / "venv" / "Scripts" / "python.exe"


def get_sa_version(py: Path):
    r = subprocess.run(
        [str(py), "-c", "from importlib.metadata import version; print(version('sageattention'))"],
        capture_output=True, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def run_interactive(base_path: Path, stdin_text: str, timeout: int = 90):
    cmd = [sys.executable, str(INSTALLER), "--install", "--sage-version", "1",
           "--base-path", str(base_path)]
    try:
        r = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                            timeout=timeout, cwd=str(REPO))
        return r.returncode, r.stdout, r.stderr, False
    except subprocess.TimeoutExpired as e:
        # The prompt is still waiting for input that never satisfies it -- this
        # itself is a finding (the process hangs rather than timing out safely).
        return None, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), \
               (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", required=True, type=Path,
                     help="ComfyUI dir with a real venv that has SageAttention 2.x installed")
    args = ap.parse_args()
    base_path = args.base_path.resolve()
    py = venv_python(base_path)

    if not py.exists():
        print(f"RESULT: FAIL setup -- no venv python at {py}")
        return 1

    scenarios = [
        ("n", "n\n"),
        ("N", "N\n"),
        ("no", "no\n"),
        ("EOF (closed stdin)", ""),
    ]

    findings = []
    for label, stdin_text in scenarios:
        before = get_sa_version(py)
        if before is None or not before.startswith("2."):
            print(f"\n[SKIP remaining] environment no longer has SA 2.x installed "
                  f"(current: {before}) -- stopping before scenario {label!r}")
            break

        print(f"\n=== Scenario: response = {label!r} ===")
        print(f"SA version before: {before}")
        rc, out, err, timed_out = run_interactive(base_path, stdin_text)
        after = get_sa_version(py)
        changed = after != before
        print(f"Exit code: {rc}  timed_out={timed_out}")
        print(f"SA version after:  {after}")
        print(f"Environment CHANGED: {changed}")
        # Show the tail of stdout around the prompt/cancellation message
        tail = out[-1500:] if out else "(no stdout captured)"
        print("--- stdout tail ---")
        print(tail)
        if err.strip():
            print("--- stderr (non-empty) ---")
            print(err[-1500:])

        expected_cancel = label in ("n", "N")
        uncaught_traceback = "Traceback (most recent call last)" in out or "Traceback (most recent call last)" in err
        if timed_out:
            findings.append(f"{label}: TIMED OUT waiting for input -- process did not "
                             f"exit safely on unsatisfiable stdin")
        elif expected_cancel and changed:
            findings.append(f"{label}: expected CANCEL but environment CHANGED "
                             f"({before} -> {after}) -- consent gate did not hold")
        elif expected_cancel and not changed:
            print(f"[OK] {label!r} correctly cancelled, environment untouched")
        elif not expected_cancel and changed:
            findings.append(f"{label}: BYPASS CONFIRMED -- response {label!r} was treated "
                             f"as consent and the environment was downgraded "
                             f"({before} -> {after})")
        elif not expected_cancel and not changed and uncaught_traceback:
            findings.append(f"{label}: did not bypass consent (environment untouched) but "
                             f"raised an UNCAUGHT exception instead of a clean cancellation "
                             f"message -- input() has no try/except around EOFError")
        elif not expected_cancel and not changed:
            print(f"[OK] {label!r} did not bypass consent (environment untouched) "
                  f"-- contrary to the source-level suspicion")

        if changed:
            # No point testing further scenarios; the env is gone.
            break

    print("\n" + "=" * 70)
    if findings:
        for f in findings:
            print(f"RESULT: FAIL {f}")
        print("OVERALL: FAIL -- consent gate has a bypass or unsafe-hang defect")
        return 1
    print("OVERALL: PASS -- consent gate held for all scenarios tested")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
