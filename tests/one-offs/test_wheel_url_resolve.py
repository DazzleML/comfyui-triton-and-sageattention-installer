"""One-off: Verify that all wheel URLs in _get_wheel_configs() resolve to real files.

Sends HTTP HEAD requests to each constructed wheel URL and reports status.
Does NOT download the wheels -- only checks they exist (HTTP 200/302).

Usage:
    python tests/one-offs/test_wheel_url_resolve.py
    python tests/one-offs/test_wheel_url_resolve.py --verbose
    python tests/one-offs/test_wheel_url_resolve.py --filter cu130  # only check cu130 wheels
"""

import argparse
import sys
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Wheel config (mirrored from comfyui_triton_sageattention.py _get_wheel_configs)
# ---------------------------------------------------------------------------
WHEEL_CONFIGS = [
    # (sage_ver, cuda, torch_pattern, py_spec, tag, is_abi3, is_experimental, torch_filename_ver)
    # === SA 2.2.0.post3 (ABI3) - STABLE ===
    ("2.2.0.post3", "130", "2.9", None, "v2.2.0-windows.post3", True, False, "2.9.0"),
    ("2.2.0.post3", "128", "2.9", None, "v2.2.0-windows.post3", True, False, "2.9.0"),
    ("2.2.0.post3", "128", "2.8", None, "v2.2.0-windows.post3", True, False, "2.8.0"),
    ("2.2.0.post3", "128", "2.7", None, "v2.2.0-windows.post3", True, False, "2.7.1"),
    ("2.2.0.post3", "126", "2.6", None, "v2.2.0-windows.post3", True, False, "2.6.0"),
    ("2.2.0.post3", "124", "2.5", None, "v2.2.0-windows.post3", True, False, "2.5.1"),
    # === SA 2.2.0.post4 (ABI3) - EXPERIMENTAL (torch 2.9) ===
    ("2.2.0.post4", "130", "2.9", None, "v2.2.0-windows.post4", True, True, "2.9.0andhigher"),
    ("2.2.0.post4", "128", "2.9", None, "v2.2.0-windows.post4", True, True, "2.9.0andhigher"),
    # === SA 2.2.0.post4 (ABI3) - PyTorch 2.10+ (forward-compat) ===
    ("2.2.0.post4", "130", "2.10", None, "v2.2.0-windows.post4", True, False, "2.9.0andhigher"),
    ("2.2.0.post4", "128", "2.10", None, "v2.2.0-windows.post4", True, False, "2.9.0andhigher"),
    # === SA 2.1.1 (per-Python) ===
    ("2.1.1", "128", "2.8.0", "313", "v2.1.1-windows", False, False, None),
    ("2.1.1", "128", "2.8.0", "312", "v2.1.1-windows", False, False, None),
    ("2.1.1", "128", "2.8.0", "311", "v2.1.1-windows", False, False, None),
    ("2.1.1", "128", "2.8.0", "310", "v2.1.1-windows", False, False, None),
    ("2.1.1", "128", "2.7.1", "312", "v2.1.1-windows", False, False, None),
    ("2.1.1", "128", "2.7.0", "312", "v2.1.1-windows", False, False, None),
    ("2.1.1", "128", "2.7.0", "311", "v2.1.1-windows", False, False, None),
    ("2.1.1", "126", "2.6.0", "312", "v2.1.1-windows", False, False, None),
    ("2.1.1", "126", "2.6.0", "311", "v2.1.1-windows", False, False, None),
    ("2.1.1", "124", "2.5.1", "312", "v2.1.1-windows", False, False, None),
    ("2.1.1", "124", "2.5.1", "311", "v2.1.1-windows", False, False, None),
    # === Legacy 2.0.1 (per-Python) ===
    ("2.0.1", "126", "2.5.0", "312", "v2.0.1-windows", False, False, None),
    ("2.0.1", "121", "2.4.0", "312", "v2.0.1-windows", False, False, None),
    ("2.0.1", "118", "2.4.0", "311", "v2.0.1-windows", False, False, None),
]


def build_wheel_url(sage_ver, cuda, torch_pattern, py_spec, tag, is_abi3, torch_filename_ver):
    """Mirror of _build_wheel_url() from comfyui_triton_sageattention.py."""
    base_url = f"https://github.com/woct0rdho/SageAttention/releases/download/{tag}"

    if is_abi3:
        sage_base = sage_ver.split(".post")[0] if ".post" in sage_ver else sage_ver
        post_suffix = ""
        if ".post" in sage_ver:
            post_idx = sage_ver.find(".post")
            post_suffix = sage_ver[post_idx:]
        if torch_filename_ver:
            torch_filename = torch_filename_ver + post_suffix
        else:
            torch_filename = torch_pattern + ".0" + post_suffix
        wheel_name = f"sageattention-{sage_base}+cu{cuda}torch{torch_filename}-cp39-abi3-win_amd64.whl"
    else:
        wheel_name = f"sageattention-{sage_ver}+cu{cuda}torch{torch_pattern}-cp{py_spec}-cp{py_spec}-win_amd64.whl"

    return f"{base_url}/{wheel_name}"


def check_url(url, timeout=15):
    """Send HEAD request, return (status_code, None) or (None, error_string)."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, None
    except urllib.error.HTTPError as e:
        return e.code, str(e)
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Verify wheel URLs resolve")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all URLs, not just failures")
    parser.add_argument("--filter", "-f", type=str, default=None, help="Only check configs matching this substring (e.g., 'cu130')")
    args = parser.parse_args()

    passed = 0
    failed = 0
    skipped = 0

    for cfg in WHEEL_CONFIGS:
        sage_ver, cuda, torch_pattern, py_spec, tag, is_abi3, is_experimental, torch_filename_ver = cfg
        label = f"SA {sage_ver} | cu{cuda} | torch {torch_pattern} | {'ABI3' if is_abi3 else f'cp{py_spec}'}"

        if args.filter and args.filter not in label and args.filter not in f"cu{cuda}":
            skipped += 1
            continue

        url = build_wheel_url(sage_ver, cuda, torch_pattern, py_spec, tag, is_abi3, torch_filename_ver)

        status, err = check_url(url)
        if status and status in (200, 302):
            passed += 1
            if args.verbose:
                print(f"  [OK]  {label}")
                print(f"        {url}")
        else:
            failed += 1
            print(f"  [FAIL] {label}  -->  HTTP {status or 'ERR'}: {err}")
            print(f"         {url}")

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        print("SOME WHEEL URLS FAILED TO RESOLVE")
        sys.exit(1)
    else:
        print("All wheel URLs verified successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
