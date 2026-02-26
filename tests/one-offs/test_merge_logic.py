"""
Test the merge-upward logic for Traffic API data.

Verifies that:
1. Math.max merge never overwrites good data with lower values
2. Expired API unique data (reported as 0) doesn't create false zeros
3. Missing uniqueClones/uniqueViews stay absent (not set to 0)
4. organicUniqueClones is skipped when uniqueClones is absent
5. Positive API unique values are always written
"""


def simulate_merge(existing_entry, clones_by_date, unique_clones_by_date,
                   views_by_date, unique_views_by_date, date_str):
    """Simulate the merge logic from traffic-badges.yml."""
    entry = dict(existing_entry)

    if date_str in clones_by_date:
        entry["clones"] = max(entry.get("clones") or 0, clones_by_date[date_str])
        api_unique_clones = unique_clones_by_date.get(date_str) or 0
        if api_unique_clones > 0:
            entry["uniqueClones"] = max(entry.get("uniqueClones") or 0, api_unique_clones)

    if date_str in views_by_date:
        entry["views"] = max(entry.get("views") or 0, views_by_date[date_str])
        api_unique_views = unique_views_by_date.get(date_str) or 0
        if api_unique_views > 0:
            entry["uniqueViews"] = max(entry.get("uniqueViews") or 0, api_unique_views)

    return entry


def simulate_organic_unique(entry):
    """Simulate organicUniqueClones calculation."""
    if entry.get("uniqueClones") is None:
        return entry  # Skip — don't create false zeros

    raw_unique = entry["uniqueClones"]
    clones = entry.get("clones") or 0
    ci_checkouts = entry.get("ciCheckouts") or 0
    ci_rate = ci_checkouts / clones if clones > 0 else 0
    ci_unique_by_pct = round(raw_unique * ci_rate)
    ci_unique_ceiling = entry.get("ciRuns") or 0
    ci_unique_clones = min(ci_unique_by_pct, ci_unique_ceiling)
    entry["organicUniqueClones"] = max(0, raw_unique - ci_unique_clones)
    return entry


def test_expired_unique_data_no_false_zeros():
    """API reports 0 for expired uniques — should NOT create false zeros."""
    existing = {"date": "2026-02-10T00:00:00Z", "clones": 26, "views": 193}
    # API reports clones=26 but uniques=0 (expired)
    result = simulate_merge(
        existing,
        clones_by_date={"2026-02-10": 26},
        unique_clones_by_date={"2026-02-10": 0},
        views_by_date={"2026-02-10": 193},
        unique_views_by_date={"2026-02-10": 0},
        date_str="2026-02-10"
    )
    assert "uniqueClones" not in result, f"uniqueClones should be absent, got {result.get('uniqueClones')}"
    assert "uniqueViews" not in result, f"uniqueViews should be absent, got {result.get('uniqueViews')}"
    print("PASS: Expired unique data does not create false zeros")


def test_positive_unique_data_written():
    """API reports positive uniques — should be written."""
    existing = {"date": "2026-02-11T00:00:00Z", "clones": 24, "views": 221}
    result = simulate_merge(
        existing,
        clones_by_date={"2026-02-11": 24},
        unique_clones_by_date={"2026-02-11": 18},
        views_by_date={"2026-02-11": 221},
        unique_views_by_date={"2026-02-11": 110},
        date_str="2026-02-11"
    )
    assert result["uniqueClones"] == 18, f"Expected 18, got {result['uniqueClones']}"
    assert result["uniqueViews"] == 110, f"Expected 110, got {result['uniqueViews']}"
    print("PASS: Positive unique data is written correctly")


def test_max_merge_never_decreases():
    """Math.max merge should never lower existing values."""
    existing = {"date": "2026-02-11T00:00:00Z", "clones": 30, "uniqueClones": 20,
                "views": 250, "uniqueViews": 120}
    # API reports lower values (stale data)
    result = simulate_merge(
        existing,
        clones_by_date={"2026-02-11": 24},
        unique_clones_by_date={"2026-02-11": 18},
        views_by_date={"2026-02-11": 200},
        unique_views_by_date={"2026-02-11": 100},
        date_str="2026-02-11"
    )
    assert result["clones"] == 30, f"Expected 30, got {result['clones']}"
    assert result["uniqueClones"] == 20, f"Expected 20, got {result['uniqueClones']}"
    assert result["views"] == 250, f"Expected 250, got {result['views']}"
    assert result["uniqueViews"] == 120, f"Expected 120, got {result['uniqueViews']}"
    print("PASS: Math.max merge never decreases existing values")


def test_max_merge_increases():
    """Math.max merge should increase when API has higher values."""
    existing = {"date": "2026-02-11T00:00:00Z", "clones": 20, "uniqueClones": 15,
                "views": 100, "uniqueViews": 50}
    result = simulate_merge(
        existing,
        clones_by_date={"2026-02-11": 30},
        unique_clones_by_date={"2026-02-11": 22},
        views_by_date={"2026-02-11": 200},
        unique_views_by_date={"2026-02-11": 80},
        date_str="2026-02-11"
    )
    assert result["clones"] == 30
    assert result["uniqueClones"] == 22
    assert result["views"] == 200
    assert result["uniqueViews"] == 80
    print("PASS: Math.max merge increases to higher API values")


def test_no_api_data_preserves_entry():
    """When date not in API at all, entry should be untouched."""
    existing = {"date": "2026-02-05T00:00:00Z", "clones": 15, "uniqueClones": 10,
                "views": 80, "uniqueViews": 40}
    result = simulate_merge(
        existing,
        clones_by_date={},
        unique_clones_by_date={},
        views_by_date={},
        unique_views_by_date={},
        date_str="2026-02-05"
    )
    assert result == existing, f"Entry should be unchanged: {result}"
    print("PASS: No API data preserves existing entry")


def test_organic_unique_skips_missing():
    """organicUniqueClones should not be computed when uniqueClones is absent."""
    entry = {"date": "2026-02-10T00:00:00Z", "clones": 26, "ciCheckouts": 0, "ciRuns": 0}
    result = simulate_organic_unique(dict(entry))
    assert "organicUniqueClones" not in result, \
        f"organicUniqueClones should be absent, got {result.get('organicUniqueClones')}"
    print("PASS: organicUniqueClones skipped when uniqueClones is missing")


def test_organic_unique_computes_with_data():
    """organicUniqueClones should compute correctly when uniqueClones is present."""
    entry = {"date": "2026-02-24T00:00:00Z", "clones": 94, "uniqueClones": 37,
             "ciCheckouts": 44, "ciRuns": 12}
    result = simulate_organic_unique(dict(entry))
    # ciRate = 44/94 ≈ 0.468
    # ciUniqueByPct = round(37 * 0.468) = round(17.3) = 17
    # ciUniqueCeiling = 12 (ciRuns)
    # ciUniqueClones = min(17, 12) = 12
    # organicUnique = max(0, 37 - 12) = 25
    assert result["organicUniqueClones"] == 25, f"Expected 25, got {result['organicUniqueClones']}"
    print("PASS: organicUniqueClones computes correctly (37 - min(17, 12) = 25)")


def test_organic_unique_zero_is_valid():
    """uniqueClones=0 is a valid value (not missing) — should compute organicUniqueClones=0."""
    entry = {"date": "2026-02-20T00:00:00Z", "clones": 5, "uniqueClones": 0,
             "ciCheckouts": 0, "ciRuns": 0}
    result = simulate_organic_unique(dict(entry))
    assert result["organicUniqueClones"] == 0, f"Expected 0, got {result['organicUniqueClones']}"
    print("PASS: uniqueClones=0 correctly produces organicUniqueClones=0")


def test_existing_with_expired_unique_preserved():
    """Entry that already has uniqueClones should NOT be overwritten by expired API zeros."""
    existing = {"date": "2026-02-11T00:00:00Z", "clones": 24, "uniqueClones": 18,
                "views": 221, "uniqueViews": 110}
    # API now reports 0 for uniques (expired after 14 days)
    result = simulate_merge(
        existing,
        clones_by_date={"2026-02-11": 24},
        unique_clones_by_date={"2026-02-11": 0},
        views_by_date={"2026-02-11": 221},
        unique_views_by_date={"2026-02-11": 0},
        date_str="2026-02-11"
    )
    assert result["uniqueClones"] == 18, f"Expected 18 preserved, got {result['uniqueClones']}"
    assert result["uniqueViews"] == 110, f"Expected 110 preserved, got {result['uniqueViews']}"
    print("PASS: Existing unique data preserved when API reports expired zeros")


if __name__ == "__main__":
    print("=== Testing merge-upward logic ===\n")
    test_expired_unique_data_no_false_zeros()
    test_positive_unique_data_written()
    test_max_merge_never_decreases()
    test_max_merge_increases()
    test_no_api_data_preserves_entry()
    test_organic_unique_skips_missing()
    test_organic_unique_computes_with_data()
    test_organic_unique_zero_is_valid()
    test_existing_with_expired_unique_preserved()
    print("\n=== All 9 tests passed ===")
