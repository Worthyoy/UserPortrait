from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from user_portrait import ProfileConfig, build_user_profiles


NOW = datetime(2026, 3, 31, 12, 0, 0)


def _row(user_id: str, action_name: str, when: datetime) -> dict[str, object]:
    return {"user_id": user_id, "action_name": action_name, "event_time": when}


def test_future_events_are_filtered_before_profile_building() -> None:
    rows = [
        _row("u1", "past_action", NOW - timedelta(days=1)),
        _row("u1", "future_action", NOW + timedelta(hours=1)),
        _row("u1", "future_action", NOW + timedelta(hours=2)),
        _row("u1", "future_action", NOW + timedelta(hours=3)),
        _row("u1", "future_action", NOW + timedelta(hours=4)),
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)

    assert "future_action" not in profiles["recent_tasks"]["action_name"].tolist()
    assert "future_action" not in profiles["periodic_tasks"]["action_name"].tolist()
    assert "future_action" not in profiles["recent_high_freq_tasks"]["action_name"].tolist()
    assert "future_action" not in profiles["frequent_tasks"]["action_name"].tolist()
    assert "future_action" not in profiles["time_period_tasks"]["action_name"].tolist()


def test_recent_tasks_use_latest_events_without_activity_window_fallback() -> None:
    rows = []
    for i in range(20):
        rows.append(_row("high_user", "daily_action", NOW - timedelta(days=i)))
    for i in [10, 11, 12, 13]:
        rows.append(_row("middle_user", f"old_action_{i}", NOW - timedelta(days=i)))

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    recent = profiles["recent_tasks"]
    middle_recent = recent[recent["user_id"] == "middle_user"]

    assert not middle_recent.empty
    assert middle_recent["rank"].tolist() == [1, 2, 3]
    assert middle_recent["action_name"].tolist() == [
        "old_action_10",
        "old_action_11",
        "old_action_12",
    ]


def test_interval_period_requires_stable_intervals() -> None:
    stable_rows = []
    for day in [1, 11, 21, 31]:
        stable_rows.append(_row("stable", "stable_interval", datetime(2026, 3, day, 9)))

    irregular_rows = []
    for day in [1, 3, 20, 31]:
        irregular_rows.append(_row("irregular", "irregular_interval", datetime(2026, 3, day, 9)))

    profiles = build_user_profiles(pd.DataFrame(stable_rows + irregular_rows), now=NOW)
    periodic = profiles["periodic_tasks"]

    stable = periodic[
        (periodic["user_id"] == "stable")
        & (periodic["action_name"] == "stable_interval")
    ]
    irregular = periodic[
        (periodic["user_id"] == "irregular")
        & (periodic["action_name"] == "irregular_interval")
    ]

    assert stable["period_type"].tolist() == ["interval"]
    assert irregular.empty


def test_daily_period_ignores_events_outside_detection_window() -> None:
    now = datetime(2026, 3, 27, 12, 0, 0)
    rows = [
        _row("u1", "daily_action", datetime(2026, 3, day, 9, 0, 0))
        for day in [1, 25, 26, 27]
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=now)
    periodic = profiles["periodic_tasks"]
    daily = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "daily_action")
    ].iloc[0]

    assert daily["period_type"] == "daily"
    assert daily["next_expected_time"] == pd.Timestamp("2026-03-28 09:00:00")


def test_daily_period_stops_when_next_expected_time_is_stale() -> None:
    now = datetime(2026, 3, 31, 12, 0, 0)
    rows = [
        _row("u1", "daily_action", datetime(2026, 3, day, 9, 0, 0))
        for day in [1, 25, 26, 27]
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=now)

    assert "daily_action" not in profiles["periodic_tasks"]["action_name"].tolist()


def test_weekly_period_ignores_events_outside_detection_window() -> None:
    rows = [
        _row("u1", "weekly_action", datetime(2025, 12, 1, 9, 0, 0)),
        _row("u1", "weekly_action", datetime(2025, 12, 2, 9, 0, 0)),
    ]
    rows.extend(
        _row("u1", "weekly_action", datetime(2026, 3, day, 9, 0, 0))
        for day in [3, 10, 17, 24, 31]
    )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    periodic = profiles["periodic_tasks"]
    weekly = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "weekly_action")
    ].iloc[0]

    assert weekly["period_type"] == "weekly"


def test_monthly_period_ignores_events_outside_detection_window() -> None:
    rows = [
        _row("u1", "monthly_action", datetime(2025, 1, 1, 9, 0, 0)),
        _row("u1", "monthly_action", datetime(2025, 1, 12, 9, 0, 0)),
    ]
    rows.extend(
        _row("u1", "monthly_action", datetime(2026, month, 25, 9, 0, 0))
        for month in [1, 2, 3]
    )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    periodic = profiles["periodic_tasks"]
    monthly = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "monthly_action")
    ].iloc[0]

    assert monthly["period_type"] == "monthly"
    assert monthly["next_expected_time"] == pd.Timestamp("2026-04-25 09:00:00")


def test_interval_period_uses_recent_occurrence_window() -> None:
    rows = [
        _row("u1", "interval_action", datetime(2025, 10, 1, 9, 0, 0)),
        _row("u1", "interval_action", datetime(2025, 10, 3, 9, 0, 0)),
    ]
    rows.extend(
        _row("u1", "interval_action", datetime(2026, 3, day, 9, 0, 0))
        for day in [1, 7, 13, 19, 25, 31]
    )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    periodic = profiles["periodic_tasks"]
    interval = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "interval_action")
    ].iloc[0]

    assert interval["period_type"] == "interval"
    assert interval["period_value"] == "every 6 days"


def test_monthly_period_takes_precedence_over_long_interval() -> None:
    rows = [
        _row("u1", "monthly_action", datetime(2026, month, 25, 9, 0, 0))
        for month in [1, 2, 3, 4]
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=datetime(2026, 4, 30, 12))
    periodic = profiles["periodic_tasks"]
    monthly = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "monthly_action")
    ].iloc[0]

    assert monthly["period_type"] == "monthly"
    assert monthly["period_value"] == "day≈25"


def test_weekly_period_tolerates_non_anchor_weekday_noise() -> None:
    rows = [
        _row("u1", "weekly_action", datetime(2026, 3, day, 9, 0, 0))
        for day in [2, 3, 9, 16, 23]
    ]

    profiles = build_user_profiles(pd.DataFrame(rows), now=datetime(2026, 3, 29, 12))
    periodic = profiles["periodic_tasks"]
    weekly = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "weekly_action")
    ].iloc[0]

    assert weekly["period_type"] == "weekly"
    assert weekly["confidence"] >= ProfileConfig().min_period_confidence


def test_monthly_period_tolerates_non_anchor_month_day_noise() -> None:
    rows = [
        _row("u1", "monthly_action", datetime(2026, month, 25, 9, 0, 0))
        for month in [1, 2, 3, 4]
    ]
    rows.append(_row("u1", "monthly_action", datetime(2026, 4, 10, 9, 0, 0)))

    profiles = build_user_profiles(pd.DataFrame(rows), now=datetime(2026, 4, 30, 12))
    periodic = profiles["periodic_tasks"]
    monthly = periodic[
        (periodic["user_id"] == "u1")
        & (periodic["action_name"] == "monthly_action")
    ].iloc[0]

    assert monthly["period_type"] == "monthly"
    assert monthly["confidence"] >= ProfileConfig().min_period_confidence


def test_interval_recent_occurrence_window_zero_does_not_use_all_history() -> None:
    rows = [
        _row("u1", "interval_action", datetime(2026, 3, day, 9, 0, 0))
        for day in [1, 7, 13, 19, 25, 31]
    ]

    profiles = build_user_profiles(
        pd.DataFrame(rows),
        now=NOW,
        config=ProfileConfig(interval_detection_recent_occurrences=0),
    )

    assert "interval_action" not in profiles["periodic_tasks"]["action_name"].tolist()


def test_non_positive_detection_windows_do_not_emit_windowed_periods() -> None:
    rows = [
        *[
            _row("daily", "daily_action", datetime(2026, 3, day, 9, 0, 0))
            for day in [29, 30, 31]
        ],
        *[
            _row("weekly", "weekly_action", datetime(2026, 3, day, 9, 0, 0))
            for day in [3, 10, 17, 24, 31]
        ],
        *[
            _row("monthly", "monthly_action", datetime(2026, month, 25, 9, 0, 0))
            for month in [1, 2, 3]
        ],
    ]

    profiles = build_user_profiles(
        pd.DataFrame(rows),
        now=NOW,
        config=ProfileConfig(
            daily_detection_window_days=0,
            weekly_detection_window_days=0,
            monthly_detection_window_days=0,
            interval_detection_recent_occurrences=0,
        ),
    )

    periodic_pairs = set(
        profiles["periodic_tasks"][["user_id", "action_name"]].itertuples(
            index=False, name=None
        )
    )
    assert ("daily", "daily_action") not in periodic_pairs
    assert ("weekly", "weekly_action") not in periodic_pairs
    assert ("monthly", "monthly_action") not in periodic_pairs


def test_users_without_recent_events_are_not_classified_as_high_activity() -> None:
    rows = [
        _row(f"inactive_{index}", "old_action", NOW - timedelta(days=100))
        for index in range(100)
    ]
    rows.extend(
        _row("active", "recent_action", NOW - timedelta(hours=index))
        for index in range(1, 11)
    )

    profiles = build_user_profiles(pd.DataFrame(rows), now=NOW)
    activity = profiles["activity_levels"]
    inactive = activity[activity["user_id"].str.startswith("inactive_")]

    assert (inactive["event_count_30d"] == 0).all()
    assert set(inactive["user_activity_level"]) == {"low"}
