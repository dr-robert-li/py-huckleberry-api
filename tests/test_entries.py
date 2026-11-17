"""Tests for generic history entry management (list with ids, edit, delete)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone

import pytest
from google.cloud.firestore import DELETE_FIELD
from pydantic import ValidationError

from huckleberry_api import HuckleberryAPI
from huckleberry_api._entries import (
    _latest_by_kind,
    _merge_changes,
    _parse_entry_id,
    _summary_payloads,
    _summary_updates,
    _validate_row,
)
from huckleberry_api.firebase_types import JsonValue
from huckleberry_api.models import HistoryRow, Tracker

NOW = 1_800_000_000.0


# ---------------------------------------------------------------------------
# Pure unit tests (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry_id", "expected"),
    [("doc1", ("doc1", None)), ("container/key-1", ("container", "key-1"))],
)
def test_parse_entry_id(entry_id: str, expected: tuple[str, str | None]) -> None:
    assert _parse_entry_id(entry_id) == expected


@pytest.mark.parametrize("entry_id", ["", "/", "a/", "/b", "a/b/c", "a//b"])
def test_parse_entry_id_rejects_malformed(entry_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid entry id"):
        _parse_entry_id(entry_id)


BOTTLE_RAW: dict[str, object] = {
    "mode": "bottle",
    "start": 100.0,
    "lastUpdated": 100.0,
    "bottleType": "Formula",
    "amount": 120.0,
    "units": "ml",
    "offset": -60.0,
    "end_offset": -60.0,
    "notes": "old",
    "appOnlyField": "kept",
}


def _merge(tracker: Tracker, raw: dict[str, object], changes: dict[str, JsonValue]):
    return _merge_changes(tracker, raw, _validate_row(tracker, raw), changes, NOW)


def test_merge_changes_applies_values_and_removals() -> None:
    merged, row = _merge("feed", BOTTLE_RAW, {"amount": 90, "notes": None})
    assert merged["amount"] == 90
    assert "notes" not in merged
    assert merged["lastUpdated"] == NOW
    # Fields the strict model ignores stay as stored.
    assert merged["appOnlyField"] == "kept"
    assert row.model_dump(by_alias=True)["amount"] == 90
    assert BOTTLE_RAW["notes"] == "old", "stored row must not be mutated"


def test_merge_changes_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown feed fields.*leftDuration"):
        _merge("feed", BOTTLE_RAW, {"leftDuration": 10})


def test_merge_changes_rejects_model_switch() -> None:
    # Keys of another row model are rejected outright...
    with pytest.raises(ValueError, match="Unknown feed fields"):
        _merge("feed", BOTTLE_RAW, {"mode": "breast", "lastSide": "left"})
    # ...and a stored extra that completes another model cannot smuggle a switch through.
    raw_with_extra = {**BOTTLE_RAW, "lastSide": "left"}
    with pytest.raises(ValueError, match="FirebaseBottleFeedIntervalData row into FirebaseBreastFeedIntervalData"):
        _merge("feed", raw_with_extra, {"mode": "breast"})


def test_merge_changes_rejects_invalid_value_and_required_removal() -> None:
    with pytest.raises(ValidationError):
        _merge("feed", BOTTLE_RAW, {"units": "gallons"})
    with pytest.raises(ValidationError):
        _merge("feed", BOTTLE_RAW, {"bottleType": None})


def test_merge_changes_allows_mode_within_same_model() -> None:
    raw: dict[str, object] = {"mode": "pee", "start": 1.0, "offset": 0.0}
    _merged, row = _merge("diaper", raw, {"mode": "poo", "color": "yellow"})
    assert row.model_dump()["mode"] == "poo"


def test_merge_changes_accepts_alias_field() -> None:
    raw: dict[str, object] = {"start": 1.0, "duration": 5.0, "offset": 0.0}
    merged, _row = _merge("sleep", raw, {"_id": "abc"})
    assert merged["_id"] == "abc"


# Each case: tracker, raw row, entry id, expected payloads (matching the writer named).
SUMMARY_CASES: list[tuple[str, Tracker, dict[str, object], str, dict[str, dict[str, object]]]] = [
    (
        "sleep",
        "sleep",
        {"start": 10, "duration": 60, "offset": -60.0, "end_offset": -60.0, "lastUpdated": 70.0},
        "s1",
        {"lastSleep": {"start": 10, "duration": 60, "offset": -60.0}},
    ),
    (
        "nursing",
        "feed",
        {
            "mode": "breast",
            "start": 10.0,
            "lastSide": "right",
            "leftDuration": 30.0,
            "rightDuration": 45.0,
            "offset": -60.0,
            "end_offset": -60.0,
            "notes": "n",
        },
        "f1",
        {
            "lastNursing": {
                "mode": "breast",
                "start": 10.0,
                "duration": 75.0,
                "leftDuration": 30.0,
                "rightDuration": 45.0,
                "offset": -60.0,
            },
            "lastSide": {"start": 10.0, "lastSide": "right"},
        },
    ),
    (
        "bottle",
        "feed",
        BOTTLE_RAW,
        "f2",
        {
            "lastBottle": {
                "mode": "bottle",
                "start": 100.0,
                "bottleType": "Formula",
                "bottleAmount": 120.0,
                "bottleUnits": "ml",
                "offset": -60.0,
            }
        },
    ),
    (
        "solids",
        "feed",
        {
            "mode": "solids",
            "start": 10.0,
            "offset": 0.0,
            "notes": "",
            "foods": {"apple": {"id": "apple", "created_name": "Apple", "source": "curated", "amount": "some"}},
        },
        "f3",
        {
            "lastSolid": {
                "mode": "solids",
                "start": 10.0,
                "offset": 0.0,
                "foods": {"apple": {"id": "apple", "created_name": "Apple", "source": "curated", "amount": "some"}},
            }
        },
    ),
    (
        "diaper",
        "diaper",
        {"mode": "both", "start": 10.0, "offset": 0.0, "color": "yellow", "notes": "x"},
        "d1",
        {"lastDiaper": {"start": 10.0, "mode": "both", "offset": 0.0}},
    ),
    (
        "potty",
        "diaper",
        {"mode": "pee", "start": 10.0, "offset": 0.0, "isPotty": True, "howItHappened": "wentPotty"},
        "d2",
        {"lastPotty": {"mode": "pee", "start": 10.0, "offset": 0.0}},
    ),
    (
        "growth",
        "health",
        {"mode": "growth", "start": 10.0, "lastUpdated": 11.0, "offset": 0.0, "weight": 5.0, "weightUnits": "kg"},
        "h1",
        {
            "lastGrowthEntry": {
                "mode": "growth",
                "start": 10.0,
                "lastUpdated": 11.0,
                "offset": 0.0,
                "weight": 5.0,
                "weightUnits": "kg",
                "_id": "h1",
                "type": "health",
                "isNight": False,
                "multientry_key": None,
            }
        },
    ),
    (
        "medication",
        "health",
        {
            "mode": "medication",
            "start": 10.0,
            "lastUpdated": 11.0,
            "offset": 0.0,
            "medication_id": "m",
            "medication_name": "Med",
            "amount": 2.5,
            "units": "ml",
            "notes": "",
        },
        "h2",
        {
            "lastMedication": {
                "mode": "medication",
                "start": 10.0,
                "lastUpdated": 11.0,
                "offset": 0.0,
                "medication_id": "m",
                "medication_name": "Med",
                "amount": 2.5,
                "units": "ml",
                "notes": "",
                "_id": "h2",
                "type": "health",
                "isNight": False,
                "multientry_key": None,
            }
        },
    ),
    (
        "batched medication without units or lastUpdated",
        "health",
        {
            "mode": "medication",
            "start": 10.0,
            "offset": 0.0,
            "medication_id": "m",
            "medication_name": "Med",
            "amount": 0.0,
            "notes": "",
        },
        "batch/k1",
        {
            "lastMedication": {
                "mode": "medication",
                "start": 10.0,
                "lastUpdated": 10.0,
                "offset": 0.0,
                "medication_id": "m",
                "medication_name": "Med",
                "amount": 0.0,
                "units": "drops",
                "notes": "",
                "_id": "k1",
                "type": "health",
                "isNight": False,
                "multientry_key": "batch",
            }
        },
    ),
    (
        "temperature",
        "health",
        {
            "mode": "temperature",
            "start": 10.0,
            "lastUpdated": 11.0,
            "offset": 0.0,
            "amount": 37.0,
            "units": "C",
            "notes": "n",
        },
        "h3",
        {
            "lastTemperature": {
                "_id": "h3",
                "type": "health",
                "mode": "temperature",
                "start": 10.0,
                "lastUpdated": 11.0,
                "offset": 0.0,
                "amount": 37.0,
                "units": "C",
                "multientry_key": None,
            }
        },
    ),
    (
        "batched temperature",
        "health",
        {"mode": "temperature", "start": 10.0, "offset": 0.0, "amount": 99.0, "units": "F"},
        "batch/k2",
        {
            "lastTemperature": {
                "_id": "k2",
                "type": "health",
                "mode": "temperature",
                "start": 10.0,
                "lastUpdated": 10.0,
                "offset": 0.0,
                "amount": 99.0,
                "units": "F",
                "multientry_key": "batch",
            }
        },
    ),
    (
        "pump",
        "pump",
        {
            "start": 10.0,
            "entryMode": "total",
            "leftAmount": 50.0,
            "rightAmount": 50.0,
            "units": "ml",
            "offset": 0.0,
            "duration": 600.0,
            "end_offset": 0.0,
            "notes": "n",
        },
        "p1",
        {
            "lastPump": {
                "start": 10.0,
                "duration": 600.0,
                "entryMode": "total",
                "leftAmount": 50.0,
                "rightAmount": 50.0,
                "units": "ml",
                "offset": 0.0,
            }
        },
    ),
    (
        "activity",
        "activities",
        {"mode": "tummyTime", "start": 10.0, "offset": 0.0, "duration": 120.0, "end_offset": 0.0, "notes": "n"},
        "a1",
        {"lastTummyTime": {"start": 10.0, "offset": 0.0, "duration": 120.0, "end_offset": 0.0}},
    ),
    (
        "activity without duration",
        "activities",
        {"mode": "bath", "start": 10.0, "offset": 0.0},
        "a2",
        {"lastBath": {"start": 10.0, "offset": 0.0}},
    ),
]


@pytest.mark.parametrize(
    ("tracker", "raw", "entry_id", "expected"),
    [case[1:] for case in SUMMARY_CASES],
    ids=[case[0] for case in SUMMARY_CASES],
)
def test_summary_payloads(
    tracker: Tracker, raw: dict[str, object], entry_id: str, expected: dict[str, dict[str, object]]
) -> None:
    row = _validate_row(tracker, raw)
    assert _summary_payloads(entry_id, row, medication_units="drops") == expected


def test_medication_summary_prefers_row_units_over_carried_units() -> None:
    raw = dict(SUMMARY_CASES[7][2])
    row = _validate_row("health", raw)
    assert _summary_payloads("h2", row, medication_units="drops")["lastMedication"]["units"] == "ml"


def _rows(tracker: Tracker, *items: tuple[str, dict[str, object]]) -> list[tuple[str, HistoryRow]]:
    return [(entry_id, _validate_row(tracker, raw)) for entry_id, raw in items]


def test_latest_by_kind_picks_latest_start_per_kind() -> None:
    rows = _rows(
        "feed",
        ("old-bottle", {**BOTTLE_RAW, "start": 50.0}),
        ("new-bottle", {**BOTTLE_RAW, "start": 150.0}),
        ("batch/mid-bottle", {**BOTTLE_RAW, "start": 120.0}),
        ("nursing", {"mode": "breast", "start": 10.0, "lastSide": "left", "offset": 0.0}),
    )
    latest = _latest_by_kind(rows)
    assert {kind: entry_id for kind, (entry_id, _row) in latest.items()} == {
        "lastBottle": "new-bottle",
        "lastNursing": "nursing",
    }


def test_latest_by_kind_separates_diaper_and_potty() -> None:
    rows = _rows(
        "diaper",
        ("diaper", {"mode": "pee", "start": 10.0, "offset": 0.0}),
        ("potty", {"mode": "pee", "start": 20.0, "offset": 0.0, "isPotty": True}),
    )
    assert set(_latest_by_kind(rows)) == {"lastDiaper", "lastPotty"}


def test_summary_updates_rewrites_found_and_removes_missing_kinds() -> None:
    latest = _latest_by_kind(_rows("feed", ("b1", BOTTLE_RAW)))
    updates = _summary_updates({"lastBottle", "lastNursing", "lastSolid"}, latest, None, NOW)
    assert updates == {
        "prefs.lastBottle": _summary_payloads("b1", latest["lastBottle"][1])["lastBottle"],
        "prefs.lastNursing": DELETE_FIELD,
        "prefs.lastSide": DELETE_FIELD,
        "prefs.lastSolid": DELETE_FIELD,
        "prefs.timestamp": {"seconds": NOW},
        "prefs.local_timestamp": NOW,
    }


# ---------------------------------------------------------------------------
# Live integration tests (skip without credentials)
# ---------------------------------------------------------------------------


async def _last_summaries(api: HuckleberryAPI, child_uid: str, tracker: Tracker) -> dict[str, object]:
    """Return the raw prefs.last* fields of a tracker document."""
    db = await api._get_firestore_client()
    snapshot = await db.collection(tracker).document(child_uid).get()
    prefs = (snapshot.to_dict() or {}).get("prefs") or {}
    return {key: value for key, value in prefs.items() if key.startswith("last")}


LogFn = Callable[[HuckleberryAPI, str, datetime], Awaitable[None]]

LIVE_CASES: list[tuple[Tracker, LogFn, dict[str, JsonValue]]] = [
    (
        "sleep",
        lambda api, cid, t: api.log_sleep(cid, start_time=t, end_time=t + timedelta(minutes=5)),
        {"duration": 600},
    ),
    (
        "feed",
        lambda api, cid, t: api.log_bottle(cid, start_time=t, amount=60.0, notes="entries test"),
        {"amount": 90.0, "notes": None},
    ),
    ("diaper", lambda api, cid, t: api.log_diaper(cid, start_time=t, mode="pee"), {"mode": "poo"}),
    (
        "health",
        lambda api, cid, t: api.log_temperature(cid, start_time=t, amount=37.0, units="C"),
        {"amount": 37.5},
    ),
    (
        "pump",
        lambda api, cid, t: api.log_pump(cid, start_time=t, duration=300, total_amount=40.0),
        {"leftAmount": 25.0},
    ),
    (
        "activities",
        lambda api, cid, t: api.log_activity(cid, mode="bath", start_time=t, duration=120),
        {"duration": 180.0},
    ),
]


@pytest.mark.parametrize(("tracker", "log", "changes"), LIVE_CASES, ids=[case[0] for case in LIVE_CASES])
async def test_entry_round_trip(
    api: HuckleberryAPI,
    child_uid: str,
    tracker: Tracker,
    log: LogFn,
    changes: dict[str, JsonValue],
) -> None:
    """Log the newest entry, find, edit, and delete it; prefs.last* must return to their prior values."""
    before = await _last_summaries(api, child_uid, tracker)
    start = datetime.now(timezone.utc).replace(microsecond=0)
    await log(api, child_uid, start)

    entry_id: str | None = None
    try:
        entries = await api.list_entries(child_uid, tracker, start, start + timedelta(seconds=1))
        matches = [entry for entry in entries if float(entry.data.start) == start.timestamp()]
        assert len(matches) == 1
        entry_id = matches[0].id
        assert await api.get_entry(child_uid, tracker, entry_id) == matches[0]

        updated = await api.update_entry(child_uid, tracker, entry_id, changes)
        stored = await api.get_entry(child_uid, tracker, entry_id)
        assert stored == updated
        dumped = updated.data.model_dump(by_alias=True)
        for key, value in changes.items():
            assert dumped[key] == value

        with pytest.raises(ValueError):
            await api.update_entry(child_uid, tracker, entry_id, {"notAField": 1})
    finally:
        if entry_id is not None:
            await api.delete_entry(child_uid, tracker, entry_id)

    assert await api.get_entry(child_uid, tracker, entry_id) is None
    with pytest.raises(ValueError):
        await api.delete_entry(child_uid, tracker, entry_id)
    assert await _last_summaries(api, child_uid, tracker) == before
