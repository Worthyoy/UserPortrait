from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta
from math import log1p
import random
from typing import TYPE_CHECKING

import pandas as pd

from .sample_data import FILLER_ACTIONS

if TYPE_CHECKING:
    from .sample_data import SampleDataConfig


PERIODS = ["early_morning", "morning", "afternoon", "evening"]
PERIOD_HOURS = {
    "early_morning": [0, 1, 5],
    "morning": [6, 9, 11],
    "afternoon": [12, 15, 17],
    "evening": [18, 21, 23],
}
PERIOD_ORDER = {name: index for index, name in enumerate(PERIODS)}
COMMON_SEQUENCE = ["公共流程起点", "公共流程终点"]


def generate_large_sample_dataset(cfg: SampleDataConfig) -> dict[str, pd.DataFrame]:
    rng = random.Random(cfg.seed)
    now = cfg.reference_time
    users = _build_large_users(cfg)
    shuffled = users.copy()
    rng.shuffle(shuffled)
    user_ids = [item["user_id"] for item in shuffled]

    events: list[dict[str, object]] = []
    periodic_truth: list[dict[str, object]] = []
    periodic_negative: list[dict[str, object]] = []
    high_freq_truth: list[dict[str, object]] = []
    high_freq_negative: list[dict[str, object]] = []
    time_period_injected: list[dict[str, object]] = []
    time_period_negative: list[dict[str, object]] = []
    sequence_truth: list[dict[str, object]] = []
    sequence_negative: list[dict[str, object]] = []

    for user in users:
        events.extend(_background_events(user, cfg, rng))

    periodic_users = user_ids[: cfg.periodic_positive_count]
    for index, user_id in enumerate(periodic_users):
        rows, truth = _periodic_positive(user_id, index, now, rng)
        events.extend(rows)
        periodic_truth.append(truth)

    negative_users = _cycle_users(user_ids[1200:], cfg.periodic_negative_count)
    for index, user_id in enumerate(negative_users):
        rows, truth = _periodic_negative(user_id, index, now, rng)
        events.extend(rows)
        periodic_negative.append(truth)

    high_freq_users = _cycle_users(user_ids[400:1200], cfg.high_freq_positive_count)
    for index, user_id in enumerate(high_freq_users):
        rows, truth = _high_freq_positive(user_id, index, now, rng)
        events.extend(rows)
        high_freq_truth.append(truth)

    high_negative_users = _cycle_users(user_ids[1200:], cfg.high_freq_negative_count)
    for index, user_id in enumerate(high_negative_users):
        rows, truth = _high_freq_negative(user_id, index, now, rng)
        events.extend(rows)
        high_freq_negative.append(truth)

    time_users = _cycle_users(user_ids[200:1200], cfg.time_period_positive_count)
    for index, user_id in enumerate(time_users):
        rows, truth = _time_period_positive(user_id, index, now, rng)
        events.extend(rows)
        time_period_injected.append(truth)

    time_negative_users = _cycle_users(user_ids[1200:], cfg.time_period_negative_count)
    for index, user_id in enumerate(time_negative_users):
        rows, truth = _time_period_negative(user_id, index, now, rng)
        events.extend(rows)
        time_period_negative.append(truth)

    sequence_users = user_ids[:600]
    for index in range(cfg.sequence_positive_count):
        user_id = sequence_users[index % len(sequence_users)]
        rows, truth = _sequence_positive(user_id, index, now)
        events.extend(rows)
        sequence_truth.append(truth)

    sequence_negative_users = _cycle_users(user_ids[1200:], cfg.sequence_negative_count)
    for index, user_id in enumerate(sequence_negative_users):
        rows, truth = _sequence_negative(user_id, index, now)
        events.extend(rows)
        sequence_negative.append(truth)

    common_user_count = int(cfg.user_count * 0.81)
    for index, user_id in enumerate(user_ids[:common_user_count]):
        events.extend(_global_common_sequence_events(user_id, index, now))

    event_frame = pd.DataFrame(events)
    event_frame["event_time"] = pd.to_datetime(event_frame["event_time"])
    event_frame = event_frame[event_frame["event_time"] <= now]
    event_frame = event_frame.sort_values(["user_id", "event_time", "action_name"])
    event_frame = event_frame.drop_duplicates(["user_id", "action_name", "event_time"])
    event_frame = event_frame.reset_index(drop=True)

    recent_truth = _recent_ground_truth(event_frame, now)
    frequent_truth = _frequent_ground_truth(event_frame)
    exhaustive_time_truth = build_expected_time_period_tasks(event_frame)
    dirty_events = _dirty_events(event_frame, users, now)

    return {
        "sample_events": event_frame,
        "sample_events_dirty": dirty_events,
        "sample_ground_truth_recent": recent_truth,
        "sample_ground_truth_frequent": frequent_truth,
        "sample_ground_truth_periodic": pd.DataFrame(periodic_truth),
        "sample_ground_truth_periodic_negative": pd.DataFrame(periodic_negative),
        "sample_ground_truth_high_freq": pd.DataFrame(high_freq_truth),
        "sample_ground_truth_high_freq_negative": pd.DataFrame(high_freq_negative),
        "sample_ground_truth_time_period": exhaustive_time_truth,
        "sample_injected_time_period_patterns": pd.DataFrame(time_period_injected),
        "sample_ground_truth_time_period_negative": pd.DataFrame(time_period_negative),
        "sample_ground_truth_sequences": pd.DataFrame(sequence_truth),
        "sample_ground_truth_sequences_negative": pd.DataFrame(sequence_negative),
    }


def build_large_dataset_manifest(
    frames: dict[str, pd.DataFrame], cfg: SampleDataConfig
) -> dict[str, object]:
    positive_tables = [
        frames["sample_ground_truth_periodic"],
        frames["sample_ground_truth_high_freq"],
        frames["sample_injected_time_period_patterns"],
        frames["sample_ground_truth_sequences"],
    ]
    user_pattern_counts: dict[str, int] = {}
    for frame in positive_tables:
        for user_id in set(frame["user_id"]):
            user_pattern_counts[str(user_id)] = user_pattern_counts.get(str(user_id), 0) + 1
    return {
        "dataset_schema_version": 2,
        "profile": "large",
        "seed": cfg.seed,
        "reference_time": cfg.reference_time.isoformat(sep=" "),
        "history_days": cfg.history_days,
        "user_count": cfg.user_count,
        "clean_event_count": len(frames["sample_events"]),
        "dirty_event_count": len(frames["sample_events_dirty"]),
        "time_period_ground_truth_scope": "exhaustive",
        "cohort_counts": {
            "high": cfg.high_users,
            "middle": cfg.middle_users,
            "low": cfg.low_users,
            "noise": cfg.noise_users,
        },
        "pattern_counts": {
            "periodic_positive": len(frames["sample_ground_truth_periodic"]),
            "periodic_negative": len(frames["sample_ground_truth_periodic_negative"]),
            "high_freq_positive": len(frames["sample_ground_truth_high_freq"]),
            "high_freq_negative": len(frames["sample_ground_truth_high_freq_negative"]),
            "time_period_injected": len(frames["sample_injected_time_period_patterns"]),
            "time_period_exhaustive": len(frames["sample_ground_truth_time_period"]),
            "time_period_negative": len(frames["sample_ground_truth_time_period_negative"]),
            "sequence_positive": len(frames["sample_ground_truth_sequences"]),
            "sequence_negative": len(frames["sample_ground_truth_sequences_negative"]),
        },
        "overlap_users": {
            "at_least_two_modules": sum(value >= 2 for value in user_pattern_counts.values()),
            "at_least_three_modules": sum(value >= 3 for value in user_pattern_counts.values()),
        },
        "dirty_input_expected": {
            "future_event_count": 50,
            "invalid_row_count": 50,
            "filtered_row_count": 100,
        },
        "file_row_counts": {name: len(frame) for name, frame in frames.items()},
    }


def build_expected_time_period_tasks(events: pd.DataFrame) -> pd.DataFrame:
    period_events = events.copy()
    period_events["time_period"] = period_events["event_time"].dt.hour.map(
        _period_label
    )
    period_events["event_date"] = period_events["event_time"].dt.date
    totals = (
        period_events.groupby(["user_id", "action_name"])
        .size()
        .rename("total_event_count")
        .reset_index()
    )
    stats = (
        period_events.groupby(["user_id", "action_name", "time_period"])
        .agg(
            period_event_count=("event_time", "size"),
            active_days=("event_date", "nunique"),
            first_time=("event_time", "min"),
            last_time=("event_time", "max"),
        )
        .reset_index()
        .merge(totals, on=["user_id", "action_name"], validate="many_to_one")
    )
    stats["period_order"] = stats["time_period"].map(PERIOD_ORDER)
    stats = stats.sort_values(
        ["user_id", "action_name", "period_event_count", "active_days", "period_order"],
        ascending=[True, True, False, False, True],
    )
    selected = stats.groupby(["user_id", "action_name"], sort=False).head(1).copy()
    selected["period_share"] = selected["period_event_count"] / selected["total_event_count"]
    selected = selected[
        (selected["period_event_count"] >= 4)
        & (selected["active_days"] >= 3)
        & (selected["period_share"] >= 0.70)
    ].copy()
    selected["time_period_score"] = selected["period_event_count"].map(log1p) * selected[
        "period_share"
    ]
    selected["period_share"] = selected["period_share"].round(6)
    selected["time_period_score"] = selected["time_period_score"].round(6)
    selected = selected.sort_values(
        [
            "user_id",
            "time_period_score",
            "period_event_count",
            "active_days",
            "last_time",
            "action_name",
        ],
        ascending=[True, False, False, False, False, True],
    )
    selected["rank"] = selected.groupby("user_id").cumcount() + 1
    selected = selected[selected["rank"] <= 5]
    return selected[
        [
            "user_id",
            "action_name",
            "time_period",
            "period_event_count",
            "total_event_count",
            "period_share",
            "active_days",
            "first_time",
            "last_time",
            "time_period_score",
            "rank",
        ]
    ].reset_index(drop=True)


def _build_large_users(cfg: SampleDataConfig) -> list[dict[str, str]]:
    users: list[dict[str, str]] = []
    index = 1
    for cohort, count in [
        ("high", cfg.high_users),
        ("middle", cfg.middle_users),
        ("low", cfg.low_users),
        ("noise", cfg.noise_users),
    ]:
        for _ in range(count):
            users.append({"user_id": f"user_{index:06d}", "cohort": cohort})
            index += 1
    return users


def _background_events(
    user: dict[str, str], cfg: SampleDataConfig, rng: random.Random
) -> list[dict[str, object]]:
    cohort = user["cohort"]
    specs = {
        "high": ((15, 22), (10, 16), (6, 10)),
        "middle": ((5, 11), (6, 11), (4, 8)),
        "low": ((0, 3), (2, 5), (2, 4)),
        "noise": ((0, 3), (4, 8), (3, 6)),
    }
    recent_count, older_count, historical_count = [
        rng.randint(*bounds) for bounds in specs[cohort]
    ]
    offsets = set(rng.sample(range(0, 30), recent_count))
    offsets.update(rng.sample(range(30, 180), older_count))
    offsets.update(rng.sample(range(180, cfg.history_days), historical_count))
    rows: list[dict[str, object]] = []
    for offset in sorted(offsets, reverse=True):
        active_date = (cfg.reference_time - timedelta(days=offset)).date()
        hour = rng.randrange(24)
        minute = rng.randrange(50)
        start = datetime.combine(active_date, time(hour, minute))
        actions = rng.sample(FILLER_ACTIONS, rng.randint(2, 3))
        rows.extend(_session(user["user_id"], actions, start, 3))
    return rows


def _periodic_positive(
    user_id: str, index: int, now: datetime, rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    period_type = ["daily", "weekly", "monthly", "interval"][index % 4]
    strength = "strong" if (index // 4) % 2 == 0 else "weak"
    action = f"大样本周期_{period_type}_{index:04d}"
    hour = [5, 6, 12, 18][index % 4]
    dates: list[date]
    if period_type == "daily":
        dates = [(now - timedelta(days=offset)).date() for offset in range(8, -1, -1)]
        period_value = "every 1 day"
        next_date = now.date() + timedelta(days=1)
    elif period_type == "weekly":
        weekday = index % 7
        last = now.date() - timedelta(days=(now.weekday() - weekday) % 7)
        dates = [last - timedelta(days=7 * step) for step in range(7, -1, -1)]
        dates.sort()
        period_value = f"weekday {weekday}"
        next_date = dates[-1] + timedelta(days=7)
    elif period_type == "monthly":
        target_day = [1, 15, 28, 30][(index // 4) % 4]
        dates = [_month_date(now.year, now.month, target_day, -offset) for offset in range(7, -1, -1)]
        period_value = f"day {target_day}"
        next_date = _month_date(now.year, now.month, target_day, 1)
    else:
        interval = [9, 11, 13, 17][(index // 4) % 4]
        last = now.date() - timedelta(days=index % 3)
        dates = [last - timedelta(days=interval * step) for step in range(7, -1, -1)]
        period_value = f"every {interval} days"
        next_date = dates[-1] + timedelta(days=interval)
    if strength == "weak" and len(dates) > 5:
        if period_type == "interval":
            dates[2] = dates[2] + timedelta(days=1)
        else:
            dates.pop(2)
            if period_type in {"weekly", "monthly"}:
                dates[2] = dates[2] + timedelta(days=1)
    rows = [
        _event(user_id, action, datetime.combine(day, time(hour, 5 + pos % 40)))
        for pos, day in enumerate(dates)
    ]
    truth = {
        "user_id": user_id,
        "action_name": action,
        "period_type": period_type,
        "period_value": period_value,
        "next_expected_time": datetime.combine(next_date, time(hour, 0)),
        "positive_label": 1,
        "pattern_id": f"large_periodic_{index:04d}",
        "pattern_strength": strength,
        "jitter_days": 1 if strength == "weak" and period_type in {"weekly", "monthly"} else 0,
        "missing_occurrence_count": (
            1 if strength == "weak" and period_type != "interval" else 0
        ),
    }
    return rows, truth


def _periodic_negative(
    user_id: str, index: int, now: datetime, rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    negative_type = [
        "two_occurrences",
        "short_burst",
        "random_intervals",
        "single_cycle",
        "stale_periodic",
    ][index % 5]
    action = f"大样本非周期_{negative_type}_{index:04d}"
    if negative_type == "two_occurrences":
        offsets = [40, 10]
    elif negative_type == "short_burst":
        offsets = [25, 24, 23, 22, 21]
    elif negative_type == "random_intervals":
        offsets = [85, 61, 44, 18, 3]
    elif negative_type == "single_cycle":
        offsets = [20, 13, 6]
    else:
        offsets = [150, 143, 136, 129, 122]
    rows = [
        _event(
            user_id,
            action,
            datetime.combine((now - timedelta(days=offset)).date(), time(10, rng.randrange(60))),
        )
        for offset in offsets
    ]
    return rows, {
        "user_id": user_id,
        "action_name": action,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"large_periodic_negative_{index:04d}",
        "reason": "should_not_be_periodic",
    }


def _high_freq_positive(
    user_id: str, index: int, now: datetime, rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    action = f"大样本近期高频_{index:04d}"
    recent_count = 4 + index % 5
    baseline_count = index % 2
    rows: list[dict[str, object]] = []
    recent_offsets = [1, 1, 2, 4, 4, 6, 6, 3]
    for occurrence in range(recent_count):
        dt = now - timedelta(
            days=recent_offsets[occurrence], hours=(occurrence * 5) % 20
        )
        rows.append(_event(user_id, action, dt.replace(minute=rng.randrange(60))))
    for occurrence in range(baseline_count):
        dt = now - timedelta(days=15 + occurrence * 8, hours=occurrence)
        rows.append(_event(user_id, action, dt.replace(minute=rng.randrange(60))))
    count_30d = recent_count + baseline_count
    lift = recent_count / max(baseline_count / 3.0, 0.5)
    return rows, {
        "user_id": user_id,
        "action_name": action,
        "recent_count": recent_count,
        "time_window": "last_7_days",
        "positive_label": 1,
        "pattern_id": f"large_high_freq_{index:04d}",
        "baseline_count": baseline_count / 3.0,
        "count_7d": recent_count,
        "count_30d": count_30d,
        "lift": round(lift, 6),
        "recent_concentration": round(recent_count / count_30d, 6),
        "reason": "recent_frequency_increase",
    }


def _high_freq_negative(
    user_id: str, index: int, now: datetime, rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    negative_type = ["low_count", "low_lift", "low_concentration", "long_term_high_freq"][
        index % 4
    ]
    action = f"大样本非近期高频_{negative_type}_{index:04d}"
    recent_count, baseline_count = {
        "low_count": (3, 0),
        "low_lift": (4, 6),
        "low_concentration": (4, 8),
        "long_term_high_freq": (6, 24),
    }[negative_type]
    rows: list[dict[str, object]] = []
    recent_offsets = [1, 1, 2, 4, 4, 6]
    for occurrence in range(recent_count):
        rows.append(
            _event(
                user_id,
                action,
                now
                - timedelta(
                    days=recent_offsets[occurrence], hours=occurrence % 12
                ),
            )
        )
    for occurrence in range(baseline_count):
        day = 8 + occurrence % 21
        hour = (occurrence * 7) % 24
        rows.append(
            _event(
                user_id,
                action,
                datetime.combine((now - timedelta(days=day)).date(), time(hour, rng.randrange(60))),
            )
        )
    count_30d = recent_count + baseline_count
    return rows, {
        "user_id": user_id,
        "action_name": action,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"large_high_freq_negative_{index:04d}",
        "baseline_count": baseline_count / 3.0,
        "count_7d": recent_count,
        "count_30d": count_30d,
        "lift": round(recent_count / max(baseline_count / 3.0, 0.5), 6),
        "recent_concentration": round(recent_count / count_30d, 6),
        "reason": "should_not_be_recent_high_frequency",
    }


def _time_period_positive(
    user_id: str, index: int, now: datetime, rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    period = PERIODS[index % 4]
    action = f"大样本固定时段_{period}_{index:04d}"
    period_count = [4, 5, 7, 8][(index // 4) % 4]
    off_count = 3 if period_count == 7 else (1 if period_count < 7 else 2)
    rows: list[dict[str, object]] = []
    irregular_offsets = [10, 31, 58, 96, 141, 199, 262, 330]
    selected_offsets = irregular_offsets[:period_count]
    if period_count == 4:
        selected_offsets[-1] = selected_offsets[-2]
    for occurrence in range(period_count):
        active_date = (
            now - timedelta(days=selected_offsets[occurrence] + index % 5)
        ).date()
        hours = PERIOD_HOURS[period]
        hour = hours[occurrence % len(hours)]
        rows.append(_event(user_id, action, datetime.combine(active_date, time(hour, rng.randrange(60)))))
    other_period = PERIODS[(index + 2) % 4]
    for occurrence in range(off_count):
        active_date = (now - timedelta(days=35 + occurrence * 19 + index % 7)).date()
        hour = PERIOD_HOURS[other_period][occurrence % 3]
        rows.append(_event(user_id, action, datetime.combine(active_date, time(hour, rng.randrange(60)))))
    total = period_count + off_count
    return rows, {
        "user_id": user_id,
        "action_name": action,
        "time_period": period,
        "period_event_count": period_count,
        "total_event_count": total,
        "period_share": round(period_count / total, 6),
        "active_days": len(set(selected_offsets)),
        "positive_label": 1,
        "pattern_id": f"large_time_period_{index:04d}",
        "reason": "fixed_time_period_high_frequency",
    }


def _time_period_negative(
    user_id: str, index: int, now: datetime, rng: random.Random
) -> tuple[list[dict[str, object]], dict[str, object]]:
    period = PERIODS[index % 4]
    negative_type = ["low_count", "low_active_days", "low_share"][index % 3]
    action = f"大样本非固定时段_{negative_type}_{index:04d}"
    if negative_type == "low_count":
        period_count, active_offsets, off_count = 3, [10, 30, 50], 0
    elif negative_type == "low_active_days":
        period_count, active_offsets, off_count = 4, [10, 10, 20, 20], 0
    else:
        period_count, active_offsets, off_count = 7, [10, 24, 43, 67, 96, 132, 175], 4
    rows: list[dict[str, object]] = []
    for occurrence, offset in enumerate(active_offsets):
        hour = PERIOD_HOURS[period][occurrence % 3]
        minute = (occurrence * 11 + rng.randrange(10)) % 60
        rows.append(
            _event(user_id, action, datetime.combine((now - timedelta(days=offset)).date(), time(hour, minute)))
        )
    other = PERIODS[(index + 1) % 4]
    for occurrence in range(off_count):
        hour = PERIOD_HOURS[other][occurrence % 3]
        rows.append(
            _event(
                user_id,
                action,
                datetime.combine((now - timedelta(days=15 + occurrence * 13)).date(), time(hour, occurrence)),
            )
        )
    return rows, {
        "user_id": user_id,
        "action_name": action,
        "time_period": period,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"large_time_period_negative_{index:04d}",
        "reason": "should_not_be_time_period_task",
    }


def _sequence_positive(
    user_id: str, index: int, now: datetime
) -> tuple[list[dict[str, object]], dict[str, object]]:
    length = 2 + index % 4
    sequence = [f"大样本链路_{index:04d}_步骤{step}" for step in range(1, length + 1)]
    repeat_count = 4 if length == 2 else 3
    gap_minutes = 30 if index % 2 == 0 else 29
    rows: list[dict[str, object]] = []
    repeat_offsets = [5, 19, 48, 83]
    for repeat in range(repeat_count):
        start = now - timedelta(
            days=repeat_offsets[repeat] + index % 11, hours=4
        )
        rows.extend(_session(user_id, sequence, start, gap_minutes))
    return rows, {
        "user_id": user_id,
        "sequence": " -> ".join(sequence),
        "sequence_length": length,
        "support_count": repeat_count,
        "next_action_candidate": sequence[-1],
        "positive_label": 1,
        "pattern_id": f"large_sequence_{index:04d}",
    }


def _sequence_negative(
    user_id: str, index: int, now: datetime
) -> tuple[list[dict[str, object]], dict[str, object]]:
    negative_type = ["low_support", "low_confidence", "session_gap", "global_common"][
        index % 4
    ]
    if negative_type == "global_common":
        return [], {
            "user_id": user_id,
            "sequence": " -> ".join(COMMON_SEQUENCE),
            "sequence_length": 2,
            "negative_type": negative_type,
            "positive_label": 0,
            "pattern_id": f"large_sequence_negative_{index:04d}",
        }
    sequence = [f"大样本负链路_{index:04d}_A", f"大样本负链路_{index:04d}_B"]
    rows: list[dict[str, object]] = []
    if negative_type == "low_support":
        rows.extend(_session(user_id, sequence, now - timedelta(days=30 + index % 20), 5))
    elif negative_type == "session_gap":
        start = now - timedelta(days=40 + index % 20)
        rows.append(_event(user_id, sequence[0], start))
        rows.append(_event(user_id, sequence[1], start + timedelta(minutes=31)))
    else:
        full_offsets = [10, 28, 61]
        for repeat in range(3):
            start = now - timedelta(days=full_offsets[repeat] + index % 9)
            rows.extend(_session(user_id, sequence, start, 5))
        for repeat in range(2):
            start = now - timedelta(days=12 + repeat * 17 + index % 9)
            rows.extend(_session(user_id, [f"负链路噪声_{index}", sequence[0]], start, 5))
    return rows, {
        "user_id": user_id,
        "sequence": " -> ".join(sequence),
        "sequence_length": 2,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"large_sequence_negative_{index:04d}",
    }


def _global_common_sequence_events(
    user_id: str, index: int, now: datetime
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    full_offsets = [7, 17, 32]
    for repeat in range(3):
        start = now - timedelta(days=full_offsets[repeat] + index % 5, hours=2)
        rows.extend(_session(user_id, COMMON_SEQUENCE, start, 4))
    prefix_offsets = [9, 24]
    for repeat in range(2):
        start = now - timedelta(days=prefix_offsets[repeat] + index % 5, hours=2)
        rows.extend(_session(user_id, [f"公共噪声_{repeat}", COMMON_SEQUENCE[0]], start, 4))
    return rows


def _recent_ground_truth(events: pd.DataFrame, now: datetime) -> pd.DataFrame:
    start = pd.Timestamp(now) - pd.Timedelta(days=30)
    recent = events[(events["event_time"] > start) & (events["event_time"] <= now)].copy()
    stats = (
        recent.assign(event_date=recent["event_time"].dt.date)
        .groupby("user_id")
        .agg(event_count_30d=("event_time", "size"), active_days_30d=("event_date", "nunique"))
        .reset_index()
    )
    all_users = pd.DataFrame({"user_id": sorted(events["user_id"].unique())})
    stats = all_users.merge(stats, on="user_id", how="left").fillna(0)
    active_counts = stats.loc[stats["event_count_30d"] > 0, "event_count_30d"]
    p80 = active_counts.quantile(0.80) if not active_counts.empty else float("inf")
    stats["expected_activity_level"] = stats.apply(
        lambda row: "low"
        if row["event_count_30d"] == 0
        else (
            "high"
            if row["active_days_30d"] >= 15 or row["event_count_30d"] >= p80
            else ("middle" if row["active_days_30d"] >= 4 else "low")
        ),
        axis=1,
    )
    levels = stats.set_index("user_id")["expected_activity_level"].to_dict()
    selected = events.sort_values(
        ["user_id", "event_time", "action_name"], ascending=[True, False, True]
    ).groupby("user_id", sort=False).head(3).copy()
    selected["expected_rank"] = selected.groupby("user_id").cumcount() + 1
    selected["expected_activity_level"] = selected["user_id"].map(levels)
    return selected[
        ["user_id", "action_name", "event_time", "expected_rank", "expected_activity_level"]
    ].reset_index(drop=True)


def _frequent_ground_truth(events: pd.DataFrame) -> pd.DataFrame:
    truth = (
        events.assign(event_date=events["event_time"].dt.date)
        .groupby(["user_id", "action_name"])
        .agg(
            expected_event_count=("event_time", "size"),
            expected_active_days=("event_date", "nunique"),
            expected_last_time=("event_time", "max"),
        )
        .reset_index()
        .sort_values(
            ["user_id", "expected_event_count", "expected_active_days", "expected_last_time", "action_name"],
            ascending=[True, False, False, False, True],
        )
    )
    truth["expected_rank"] = truth.groupby("user_id").cumcount() + 1
    truth = truth[truth["expected_rank"] <= 5]
    return truth[
        ["user_id", "action_name", "expected_event_count", "expected_active_days", "expected_rank"]
    ].reset_index(drop=True)


def _dirty_events(
    clean: pd.DataFrame, users: list[dict[str, str]], now: datetime
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(50):
        rows.append(
            _event(
                users[index]["user_id"],
                f"未来事件_{index:03d}",
                now + timedelta(days=1 + index % 10),
            )
        )
    for index in range(20):
        rows.append(
            {"user_id": users[index]["user_id"], "action_name": "非法时间", "event_time": "not-a-time"}
        )
    for index in range(15):
        rows.append({"user_id": None, "action_name": "空用户", "event_time": now - timedelta(days=index)})
    for index in range(15):
        rows.append(
            {"user_id": users[index]["user_id"], "action_name": None, "event_time": now - timedelta(days=index)}
        )
    dirty = clean.assign(event_time=clean["event_time"].astype(str))
    return pd.concat([dirty, pd.DataFrame(rows)], ignore_index=True)


def _period_label(hour: int) -> str:
    return PERIODS[min(hour // 6, 3)]


def _cycle_users(users: list[str], count: int) -> list[str]:
    return [users[index % len(users)] for index in range(count)]


def _event(user_id: str, action_name: str, event_time: datetime) -> dict[str, object]:
    return {"user_id": user_id, "action_name": action_name, "event_time": event_time}


def _session(
    user_id: str, actions: list[str], start: datetime, gap_minutes: int
) -> list[dict[str, object]]:
    return [
        _event(user_id, action, start + timedelta(minutes=gap_minutes * position))
        for position, action in enumerate(actions)
    ]


def _month_date(year: int, month: int, target_day: int, offset: int) -> date:
    month_index = year * 12 + month - 1 + offset
    target_year, target_month_zero = divmod(month_index, 12)
    target_month = target_month_zero + 1
    day = min(target_day, calendar.monthrange(target_year, target_month)[1])
    return date(target_year, target_month, day)
