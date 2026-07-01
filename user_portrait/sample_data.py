from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


NOW = datetime(2026, 3, 31, 12, 0, 0)
START_DATE = date(2026, 1, 1)

ACTION_BALANCE = "\u67e5\u4f59\u989d"
ACTION_BILL = "\u67e5\u8d26\u5355"
ACTION_TRANSFER = "\u7ed9\u670b\u53cb\u8f6c\u8d26"
ACTION_TRANSFER_RESULT = "\u67e5\u4ea4\u6613\u7ed3\u679c"
ACTION_CREDIT_CARD = "\u4fe1\u7528\u5361\u8d26\u5355"
ACTION_REPAY_CARD = "\u8fd8\u4fe1\u7528\u5361"
ACTION_REPAY_RESULT = "\u67e5\u8fd8\u6b3e\u7ed3\u679c"
ACTION_VIEW_FINANCE = "\u67e5\u770b\u7406\u8d22\u4ea7\u54c1"
ACTION_RISK = "\u98ce\u9669\u8bc4\u4f30"
ACTION_BUY_FINANCE = "\u8d2d\u4e70\u7406\u8d22"
ACTION_BUY_GOLD = "\u4e70\u9ec4\u91d1"
ACTION_VIEW_GOLD_MARKET = "\u67e5\u770b\u9ec4\u91d1\u884c\u60c5"
ACTION_CARD_MANAGE = "\u94f6\u884c\u5361\u7ba1\u7406"
ACTION_ADD_CARD = "\u6dfb\u52a0\u94f6\u884c\u5361"
ACTION_SMS = "\u77ed\u4fe1\u9a8c\u8bc1"
ACTION_FUND = "\u57fa\u91d1\u5b9a\u6295"
ACTION_VIEW_FUND_NAV = "\u67e5\u770b\u57fa\u91d1\u51c0\u503c"
ACTION_SALARY = "\u5de5\u8d44\u8f6c\u8d26"

PERIODIC_ACTIONS = [
    ACTION_BALANCE,
    ACTION_BILL,
    ACTION_TRANSFER,
    ACTION_CREDIT_CARD,
    ACTION_REPAY_CARD,
    ACTION_FUND,
    ACTION_SALARY,
    ACTION_BUY_FINANCE,
]

HIGH_FREQ_ACTIONS = [
    "\u4fee\u6539\u8f6c\u8d26\u9650\u989d",
    "\u5f00\u901a\u5feb\u6377\u652f\u4ed8",
    "\u4e0a\u4f20\u6536\u5165\u8bc1\u660e",
    "\u7533\u8bf7\u8d37\u6b3e\u8bd5\u7b97",
    "\u8d2d\u4e70\u65c5\u884c\u4fdd\u9669",
    "\u67e5\u8be2\u6c47\u7387\u724c\u4ef7",
]

TIME_PERIOD_ACTIONS = [
    "\u67e5\u770b\u5de5\u8d44\u6d41\u6c34",
    "\u67e5\u770b\u516c\u79ef\u91d1",
    "\u67e5\u770b\u793e\u4fdd",
    "\u4e2a\u7a0e\u8bd5\u7b97",
    "\u6c47\u6b3e\u8ddf\u8e2a",
    "\u4e0b\u8f7d\u6d41\u6c34",
]

FILLER_ACTIONS = [
    ACTION_TRANSFER_RESULT,
    ACTION_REPAY_RESULT,
    ACTION_VIEW_FINANCE,
    ACTION_RISK,
    ACTION_BUY_GOLD,
    ACTION_VIEW_GOLD_MARKET,
    ACTION_CARD_MANAGE,
    ACTION_ADD_CARD,
    ACTION_SMS,
    ACTION_VIEW_FUND_NAV,
    "\u67e5\u94f6\u884c\u5361",
    "\u67e5\u8d26\u6237\u660e\u7ec6",
    "\u67e5\u4ea4\u6613\u660e\u7ec6",
    "\u9884\u7ea6\u7f51\u70b9\u53d6\u53f7",
    "\u5feb\u6377\u652f\u4ed8",
    "\u8de8\u884c\u8f6c\u5165",
    "\u6536\u6b3e\u7801",
    "\u7ed3\u6784\u6027\u5b58\u6b3e",
]

SEQUENCE_TEMPLATES = [
    [ACTION_BALANCE, ACTION_TRANSFER, ACTION_TRANSFER_RESULT],
    [ACTION_VIEW_FINANCE, ACTION_RISK, ACTION_BUY_FINANCE],
    [ACTION_CREDIT_CARD, ACTION_REPAY_CARD, ACTION_REPAY_RESULT],
    [ACTION_CARD_MANAGE, ACTION_ADD_CARD, ACTION_SMS],
    [ACTION_FUND, ACTION_VIEW_FUND_NAV],
    [ACTION_VIEW_FINANCE, ACTION_RISK, ACTION_BUY_FINANCE, "\u67e5\u4ea4\u6613\u7ed3\u679c"],
    [ACTION_BILL, ACTION_CREDIT_CARD, ACTION_REPAY_CARD, ACTION_REPAY_RESULT],
    ["\u9009\u62e9\u7f51\u70b9", "\u9884\u7ea6\u53d6\u53f7", "\u67e5\u9884\u7ea6\u8bb0\u5f55"],
    [ACTION_BALANCE, ACTION_BILL, ACTION_CREDIT_CARD, ACTION_REPAY_CARD, ACTION_REPAY_RESULT],
    [ACTION_VIEW_GOLD_MARKET, ACTION_BUY_GOLD],
]


@dataclass(frozen=True)
class SampleDataConfig:
    profile: str = "baseline"
    seed: int = 20260624
    high_users: int = 60
    middle_users: int = 80
    low_users: int = 70
    noise_users: int = 30
    reference_time: datetime = NOW
    history_days: int = 90
    periodic_positive_count: int = 140
    periodic_negative_count: int = 180
    high_freq_positive_count: int = 100
    high_freq_negative_count: int = 120
    time_period_positive_count: int = 80
    time_period_negative_count: int = 60
    sequence_positive_count: int = 160
    sequence_negative_count: int = 0

    @property
    def user_count(self) -> int:
        return self.high_users + self.middle_users + self.low_users + self.noise_users

    @classmethod
    def for_profile(cls, profile: str, *, seed: int | None = None) -> "SampleDataConfig":
        if profile == "baseline":
            return cls(seed=seed if seed is not None else 20260624)
        if profile != "large":
            raise ValueError(f"unknown sample profile: {profile}")
        return cls(
            profile="large",
            seed=seed if seed is not None else 20260701,
            high_users=500,
            middle_users=650,
            low_users=600,
            noise_users=250,
            reference_time=datetime(2026, 6, 30, 12, 0, 0),
            history_days=365,
            periodic_positive_count=1000,
            periodic_negative_count=800,
            high_freq_positive_count=800,
            high_freq_negative_count=800,
            time_period_positive_count=1000,
            time_period_negative_count=800,
            sequence_positive_count=1200,
            sequence_negative_count=600,
        )


def generate_sample_dataset(
    config: SampleDataConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or SampleDataConfig()
    if cfg.profile == "large":
        from .large_sample_data import generate_large_sample_dataset

        return generate_large_sample_dataset(cfg)
    if cfg.profile != "baseline":
        raise ValueError(f"unknown sample profile: {cfg.profile}")
    rng = random.Random(cfg.seed)
    users = _build_users(cfg)
    shuffled_users = users.copy()
    rng.shuffle(shuffled_users)

    events: list[dict[str, object]] = []
    periodic_truth: list[dict[str, object]] = []
    periodic_negative_truth: list[dict[str, object]] = []
    high_freq_truth: list[dict[str, object]] = []
    high_freq_negative_truth: list[dict[str, object]] = []
    time_period_truth: list[dict[str, object]] = []
    time_period_negative_truth: list[dict[str, object]] = []
    sequence_truth: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()

    for user in users:
        events.extend(_generate_background_events(user, rng))

    pure_periodic_users = shuffled_users[:50]
    pure_high_freq_users = shuffled_users[50:90]
    pure_sequence_users = shuffled_users[90:130]
    mixed_users = shuffled_users[130:210]
    ordinary_users = shuffled_users[210:240]

    periodic_user_pool = pure_periodic_users + mixed_users
    high_freq_user_pool = pure_high_freq_users + mixed_users
    sequence_user_pool = pure_sequence_users + mixed_users[:40]
    negative_user_pool = ordinary_users + mixed_users + pure_periodic_users + pure_high_freq_users

    for idx in range(140):
        user = periodic_user_pool[idx % len(periodic_user_pool)]
        pattern = _periodic_pattern_for_index(idx)
        pattern = {
            **pattern,
            "action_name": _select_unused_action(
                user["user_id"], PERIODIC_ACTIONS, used_pairs, rng
            ),
        }
        strength = "strong" if idx < 90 else "weak"
        inserted, truth = _generate_periodic_events(
            user["user_id"], pattern, idx, rng, strength=strength
        )
        events.extend(inserted)
        periodic_truth.append(truth)
        used_pairs.add((user["user_id"], str(pattern["action_name"])))

    for idx in range(100):
        user = high_freq_user_pool[idx % len(high_freq_user_pool)]
        action = _select_unused_action(user["user_id"], HIGH_FREQ_ACTIONS, used_pairs, rng)
        inserted, truth = _generate_high_frequency_events(user["user_id"], action, idx, rng)
        events.extend(inserted)
        high_freq_truth.append(truth)
        used_pairs.add((user["user_id"], action))

    for idx, user in enumerate(sequence_user_pool):
        selected = [
            SEQUENCE_TEMPLATES[idx % len(SEQUENCE_TEMPLATES)],
            SEQUENCE_TEMPLATES[(idx + 3) % len(SEQUENCE_TEMPLATES)],
        ]
        for seq_idx, sequence in enumerate(selected):
            inserted, truth = _generate_sequence_events(
                user["user_id"], sequence, idx, seq_idx, rng
            )
            events.extend(inserted)
            sequence_truth.append(truth)

    for idx in range(180):
        user = negative_user_pool[idx % len(negative_user_pool)]
        action = _select_unused_action(
            user["user_id"], PERIODIC_ACTIONS + FILLER_ACTIONS, used_pairs, rng
        )
        inserted, truth = _generate_periodic_negative_events(
            user["user_id"], action, idx, rng
        )
        events.extend(inserted)
        periodic_negative_truth.append(truth)
        used_pairs.add((user["user_id"], action))

    for idx in range(120):
        if idx < 25 and periodic_truth:
            truth = _high_freq_negative_from_periodic(periodic_truth[idx], idx)
            high_freq_negative_truth.append(truth)
            continue
        user = negative_user_pool[(idx * 3) % len(negative_user_pool)]
        action = _select_unused_action(user["user_id"], HIGH_FREQ_ACTIONS, used_pairs, rng)
        inserted, truth = _generate_high_frequency_negative_events(
            user["user_id"], action, idx, rng
        )
        events.extend(inserted)
        high_freq_negative_truth.append(truth)
        used_pairs.add((user["user_id"], action))

    for idx in range(80):
        user = shuffled_users[(idx * 5) % len(shuffled_users)]
        action = _select_unused_action(user["user_id"], TIME_PERIOD_ACTIONS, used_pairs, rng)
        inserted, truth = _generate_time_period_events(user["user_id"], action, idx, rng)
        events.extend(inserted)
        time_period_truth.append(truth)
        used_pairs.add((user["user_id"], action))

    for idx in range(60):
        user = negative_user_pool[(idx * 7) % len(negative_user_pool)]
        action = _select_unused_action(user["user_id"], TIME_PERIOD_ACTIONS, used_pairs, rng)
        inserted, truth = _generate_time_period_negative_events(
            user["user_id"], action, idx, rng
        )
        events.extend(inserted)
        time_period_negative_truth.append(truth)
        used_pairs.add((user["user_id"], action))

    event_frame = pd.DataFrame(events)
    event_frame = event_frame[event_frame["event_time"] <= NOW]
    event_frame = event_frame.sort_values(["user_id", "event_time", "action_name"])
    event_frame = event_frame.drop_duplicates(["user_id", "action_name", "event_time"])
    event_frame = event_frame.reset_index(drop=True)

    recent_truth = _build_recent_ground_truth(event_frame)
    frequent_truth = _build_frequent_ground_truth(event_frame)
    return {
        "sample_events": event_frame,
        "sample_ground_truth_recent": recent_truth,
        "sample_ground_truth_frequent": frequent_truth,
        "sample_ground_truth_periodic": pd.DataFrame(periodic_truth),
        "sample_ground_truth_periodic_negative": pd.DataFrame(periodic_negative_truth),
        "sample_ground_truth_high_freq": pd.DataFrame(high_freq_truth),
        "sample_ground_truth_high_freq_negative": pd.DataFrame(high_freq_negative_truth),
        "sample_ground_truth_time_period": pd.DataFrame(time_period_truth),
        "sample_ground_truth_time_period_negative": pd.DataFrame(
            time_period_negative_truth
        ),
        "sample_ground_truth_sequences": pd.DataFrame(sequence_truth),
    }


def write_sample_dataset(
    output_dir: str | Path,
    config: SampleDataConfig | None = None,
    *,
    encoding: str = "utf-8-sig",
) -> dict[str, Path]:
    cfg = config or SampleDataConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frames = generate_sample_dataset(cfg)
    written: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output_path / f"{name}.csv"
        frame.to_csv(path, index=False, encoding=encoding)
        written[name] = path
    if cfg.profile == "large":
        from .large_sample_data import build_large_dataset_manifest

        manifest_path = output_path / "sample_dataset_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                build_large_dataset_manifest(frames, cfg),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        written["sample_dataset_manifest"] = manifest_path
    return written


def _build_users(cfg: SampleDataConfig) -> list[dict[str, str]]:
    users: list[dict[str, str]] = []
    specs = [
        ("high", cfg.high_users),
        ("middle", cfg.middle_users),
        ("low", cfg.low_users),
        ("noise", cfg.noise_users),
    ]
    index = 1
    for group, count in specs:
        for _ in range(count):
            users.append({"user_id": f"user_{index:03d}", "cohort": group})
            index += 1
    return users


def _select_unused_action(
    user_id: str,
    candidates: list[str],
    used_pairs: set[tuple[str, str]],
    rng: random.Random,
) -> str:
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    for action in shuffled:
        if (user_id, action) not in used_pairs:
            return action
    return shuffled[0]


def _generate_background_events(user: dict[str, str], rng: random.Random) -> list[dict[str, object]]:
    user_id = user["user_id"]
    cohort = user["cohort"]
    if cohort == "high":
        recent_days = rng.sample(range(0, 30), rng.randint(15, 20))
        older_days = rng.sample(range(30, 90), rng.randint(6, 10))
        sessions_per_day = (1, 2)
    elif cohort == "middle":
        recent_days = rng.sample(range(0, 30), rng.randint(5, 10))
        older_days = rng.sample(range(30, 90), rng.randint(3, 7))
        sessions_per_day = (1, 1)
    elif cohort == "low":
        recent_days = rng.sample(range(0, 30), rng.randint(0, 2))
        older_days = rng.sample(range(30, 90), rng.randint(0, 3))
        sessions_per_day = (1, 1)
    else:
        recent_days = rng.sample(range(0, 30), rng.randint(0, 3))
        older_days = rng.sample(range(30, 90), rng.randint(4, 8))
        sessions_per_day = (1, 1)

    rows: list[dict[str, object]] = []
    for offset in sorted(set(recent_days + older_days), reverse=True):
        active_date = (NOW - timedelta(days=offset)).date()
        for _ in range(rng.randint(*sessions_per_day)):
            start = _random_time_on(active_date, rng)
            actions = rng.sample(FILLER_ACTIONS, rng.randint(2, 4))
            rows.extend(_session_rows(user_id, actions, start, rng))
    return rows


def _periodic_pattern_for_index(index: int) -> dict[str, object]:
    pattern_index = index % 90
    if pattern_index < 24:
        return {
            "period_type": "daily",
            "action_name": PERIODIC_ACTIONS[pattern_index % len(PERIODIC_ACTIONS)],
            "hour": 8 + index % 4,
            "start_offset": 18 + index % 20,
        }
    if pattern_index < 48:
        weekday = index % 7
        return {
            "period_type": "weekly",
            "action_name": PERIODIC_ACTIONS[pattern_index % len(PERIODIC_ACTIONS)],
            "weekday": weekday,
            "hour": 8 + index % 4,
        }
    if pattern_index < 70:
        day = [10, 15, 25][index % 3]
        return {
            "period_type": "monthly",
            "action_name": PERIODIC_ACTIONS[pattern_index % len(PERIODIC_ACTIONS)],
            "day": day,
            "hour": 8 + index % 4,
        }
    interval_days = [14, 30][index % 2]
    return {
        "period_type": "interval",
        "action_name": PERIODIC_ACTIONS[pattern_index % len(PERIODIC_ACTIONS)],
        "interval_days": interval_days,
        "hour": 8 + index % 4,
    }


def _generate_periodic_events(
    user_id: str,
    pattern: dict[str, object],
    index: int,
    rng: random.Random,
    *,
    strength: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    period_type = str(pattern["period_type"])
    action = str(pattern["action_name"])
    hour = int(pattern["hour"])
    dates: list[date]
    jitter_days = 0
    missing_count = 0

    if period_type == "daily":
        start_offset = int(pattern["start_offset"])
        dates = [(NOW - timedelta(days=offset)).date() for offset in range(start_offset, -1, -1)]
        period_value = "every 1 day"
    elif period_type == "weekly":
        weekday = int(pattern["weekday"])
        dates = [
            START_DATE + timedelta(days=day_offset)
            for day_offset in range((NOW.date() - START_DATE).days + 1)
            if (START_DATE + timedelta(days=day_offset)).weekday() == weekday
        ][-10:]
        period_value = f"weekday={weekday}"
    elif period_type == "monthly":
        target_day = int(pattern["day"])
        dates = [
            date(2026, 1, target_day),
            date(2026, 2, min(target_day, 28)),
            date(2026, 3, target_day),
        ]
        period_value = f"day~{target_day}"
    else:
        interval_days = int(pattern["interval_days"])
        if interval_days == 30:
            dates = [
                date(2026, 1, 1),
                date(2026, 1, 31),
                date(2026, 3, 2),
                date(2026, 3, 31),
            ]
        else:
            count = 6
            first = NOW.date() - timedelta(days=interval_days * (count - 1))
            dates = [first + timedelta(days=interval_days * step) for step in range(count)]
        period_value = f"every {interval_days} days"

    if strength == "weak":
        dates, jitter_days, missing_count = _weaken_period_dates(
            dates, period_type, rng, pattern
        )

    rows = []
    for pos, d in enumerate(dates):
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(d, time(hour, (pos * 7 + index) % 55)),
            }
        )
    if strength == "weak" and period_type == "daily" and len(dates) >= 6:
        duplicate_dates = rng.sample(dates[2:-2], min(2, len(dates[2:-2])))
        for d in duplicate_dates:
            rows.append(
                {
                    "user_id": user_id,
                    "action_name": action,
                    "event_time": datetime.combine(d, time(min(hour + 1, 11))),
                }
            )

    next_time = _next_period_time(dates[-1], pattern)
    truth = {
        "user_id": user_id,
        "action_name": action,
        "period_type": period_type,
        "period_value": period_value,
        "next_expected_time": next_time,
        "positive_label": 1,
        "pattern_id": f"periodic_{index:03d}",
        "pattern_strength": strength,
        "jitter_days": jitter_days,
        "missing_occurrence_count": missing_count,
    }
    return rows, truth


def _weaken_period_dates(
    dates: list[date],
    period_type: str,
    rng: random.Random,
    pattern: dict[str, object],
) -> tuple[list[date], int, int]:
    if len(dates) <= 3:
        return dates, 0, 0

    missing_count = 0
    jitter_days = 0
    weakened = dates.copy()
    if period_type == "daily":
        missing_count = rng.randint(1, min(3, max(1, len(weakened) // 8)))
        remove_indexes = set(rng.sample(range(2, len(weakened) - 2), missing_count))
        weakened = [item for pos, item in enumerate(weakened) if pos not in remove_indexes]
    elif period_type == "weekly":
        jitter_days = 1
        jitter_count = max(1, min(2, len(weakened) // 5))
        for pos in rng.sample(range(1, len(weakened) - 1), jitter_count):
            weakened[pos] = weakened[pos] + timedelta(days=rng.choice([-1, 1]))
    elif period_type == "monthly":
        jitter_days = 2
        target_day = int(pattern["day"])
        month_dates: list[date] = []
        for original in weakened:
            drift = rng.choice([-2, -1, 1, 2])
            days_in_month = pd.Timestamp(
                year=original.year, month=original.month, day=1
            ).days_in_month
            day = max(1, min(days_in_month, target_day + drift))
            month_dates.append(date(original.year, original.month, day))
        weakened = month_dates
    else:
        jitter_days = 1
        for pos in range(1, len(weakened) - 1):
            if pos % 3 == 0:
                weakened[pos] = weakened[pos] + timedelta(days=rng.choice([-1, 1]))

    weakened = sorted(d for d in weakened if d <= NOW.date())
    return weakened, jitter_days, missing_count


def _next_period_time(last_date: date, pattern: dict[str, object]) -> datetime:
    hour = int(pattern["hour"])
    period_type = str(pattern["period_type"])
    if period_type == "daily":
        next_date = last_date + timedelta(days=1)
    elif period_type == "weekly":
        next_date = last_date + timedelta(days=7)
    elif period_type == "monthly":
        month = last_date.month + 1
        year = last_date.year
        if month > 12:
            year += 1
            month = 1
        target_day = int(pattern["day"])
        days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
        next_date = date(year, month, min(target_day, days_in_month))
    else:
        next_date = last_date + timedelta(days=int(pattern["interval_days"]))
    return datetime.combine(next_date, time(hour))


def _generate_high_frequency_events(
    user_id: str,
    action: str,
    index: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    count = 4 + index % 5
    recent_offsets = [1, 2, 3, 5]
    rows: list[dict[str, object]] = []
    for occurrence in range(count):
        day = (NOW - timedelta(days=recent_offsets[occurrence % len(recent_offsets)])).date()
        event_time = datetime.combine(
            day,
            time(9 + (occurrence * 2 + index) % 10, rng.randint(0, 55)),
        )
        rows.append({"user_id": user_id, "action_name": action, "event_time": event_time})

    baseline_events = index % 3
    for occurrence in range(baseline_events):
        day = (NOW - timedelta(days=30 + 17 * occurrence + index % 5)).date()
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(day, time(10 + occurrence, rng.randint(0, 55))),
            }
        )

    count_7d = count
    count_30d = count
    baseline_count = baseline_events / ((90 - 7) / 7)
    lift = count_7d / max(baseline_count, 0.5)
    recent_concentration = count_7d / max(count_30d, 1)
    truth = {
        "user_id": user_id,
        "action_name": action,
        "recent_count": count,
        "time_window": "7d",
        "positive_label": 1,
        "pattern_id": f"high_freq_{index:03d}",
        "baseline_count": round(baseline_count, 6),
        "count_7d": count_7d,
        "count_30d": count_30d,
        "lift": round(lift, 6),
        "recent_concentration": round(recent_concentration, 6),
        "reason": "recent_7d_lift",
    }
    return rows, truth


def _generate_periodic_negative_events(
    user_id: str,
    action: str,
    index: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    negative_type = [
        "only_two_occurrences",
        "short_burst",
        "irregular_interval",
        "single_natural_cycle",
        "frequent_without_stable_interval",
    ][index % 5]
    rows: list[dict[str, object]] = []

    if negative_type == "only_two_occurrences":
        dates = [date(2026, 3, 3), date(2026, 3, 24)]
    elif negative_type == "short_burst":
        start = NOW.date() - timedelta(days=20 + index % 15)
        dates = [start + timedelta(days=step) for step in range(3 + index % 4)]
    elif negative_type == "irregular_interval":
        offsets = sorted(rng.sample(range(5, 85), 5 + index % 2), reverse=True)
        dates = [(NOW - timedelta(days=offset)).date() for offset in offsets]
    elif negative_type == "single_natural_cycle":
        if index % 2 == 0:
            dates = [date(2026, 3, 2), date(2026, 3, 9)]
        else:
            dates = [date(2026, 2, 25), date(2026, 3, 25)]
    else:
        offsets = [3, 4, 9, 17, 28, 46, 73]
        dates = [(NOW - timedelta(days=offset + index % 3)).date() for offset in offsets]

    for pos, d in enumerate(sorted(set(dates))):
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(d, time(13 + pos % 6, rng.randint(0, 55))),
            }
        )

    truth = {
        "user_id": user_id,
        "action_name": action,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"periodic_negative_{index:03d}",
        "reason": "should_not_be_periodic",
    }
    return rows, truth


def _generate_high_frequency_negative_events(
    user_id: str,
    action: str,
    index: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    negative_type = [
        "long_term_high_frequency",
        "low_count_high_lift",
        "low_recent_concentration",
    ][index % 3]
    rows: list[dict[str, object]] = []

    if negative_type == "long_term_high_frequency":
        recent_offsets = [1, 2, 4, 6]
        older_offsets = [8, 10, 12, 15, 17, 20, 23, 26, 29]
        history_offsets = [35, 42, 49, 56, 63, 70, 77, 84]
    elif negative_type == "low_count_high_lift":
        recent_offsets = [1, 3]
        older_offsets = []
        history_offsets = []
    else:
        recent_offsets = [1, 2, 4, 6]
        older_offsets = [8, 9, 11, 13, 15, 18, 21, 24, 27, 29]
        history_offsets = []

    occurrence = 0
    for offset in recent_offsets + older_offsets + history_offsets:
        day = (NOW - timedelta(days=offset + (index % 2))).date()
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(
                    day, time(8 + occurrence % 12, rng.randint(0, 55))
                ),
            }
        )
        occurrence += 1

    count_7d = len(recent_offsets)
    count_30d = len(recent_offsets) + len(older_offsets)
    baseline_count = len(history_offsets) / ((90 - 7) / 7)
    lift = count_7d / max(baseline_count, 0.5)
    recent_concentration = count_7d / max(count_30d, 1)
    truth = {
        "user_id": user_id,
        "action_name": action,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"high_freq_negative_{index:03d}",
        "baseline_count": round(baseline_count, 6),
        "count_7d": count_7d,
        "count_30d": count_30d,
        "lift": round(lift, 6),
        "recent_concentration": round(recent_concentration, 6),
        "reason": "should_not_be_high_frequency",
    }
    return rows, truth


def _high_freq_negative_from_periodic(
    periodic_truth: dict[str, object], index: int
) -> dict[str, object]:
    return {
        "user_id": periodic_truth["user_id"],
        "action_name": periodic_truth["action_name"],
        "negative_type": "already_periodic",
        "positive_label": 0,
        "pattern_id": f"high_freq_negative_periodic_{index:03d}",
        "baseline_count": "",
        "count_7d": "",
        "count_30d": "",
        "lift": "",
        "recent_concentration": "",
        "reason": "periodic_task_should_not_repeat_as_high_frequency",
    }


def _generate_time_period_events(
    user_id: str,
    action: str,
    index: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    time_period = ["early_morning", "morning", "afternoon", "evening"][index % 4]
    offsets = [83, 64, 41, 22, 11, 4]
    period_count = 4 + index % 3
    rows: list[dict[str, object]] = []
    for occurrence, offset in enumerate(offsets[:period_count]):
        active_date = (NOW - timedelta(days=offset + (index % 5))).date()
        hour = _hour_for_time_period(time_period, occurrence)
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(active_date, time(hour, rng.randint(0, 55))),
            }
        )

    off_period_count = 1 if period_count <= 5 else 2
    for occurrence in range(off_period_count):
        active_date = (NOW - timedelta(days=70 - occurrence * 17 - index % 3)).date()
        off_period = _different_time_period(time_period)
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(
                    active_date,
                    time(_hour_for_time_period(off_period, occurrence), rng.randint(0, 55)),
                ),
            }
        )

    total_count = period_count + off_period_count
    truth = {
        "user_id": user_id,
        "action_name": action,
        "time_period": time_period,
        "period_event_count": period_count,
        "total_event_count": total_count,
        "period_share": round(period_count / total_count, 6),
        "active_days": period_count,
        "positive_label": 1,
        "pattern_id": f"time_period_{index:03d}",
        "reason": "fixed_time_period_high_frequency",
    }
    return rows, truth


def _generate_time_period_negative_events(
    user_id: str,
    action: str,
    index: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    negative_type = ["low_count", "low_active_days", "low_share"][index % 3]
    time_period = ["early_morning", "morning", "afternoon", "evening"][index % 4]
    rows: list[dict[str, object]] = []

    if negative_type == "low_count":
        offsets = [78, 39, 12]
        period_offsets = offsets
        off_offsets: list[int] = []
    elif negative_type == "low_active_days":
        period_offsets = [50, 50, 19, 19]
        off_offsets = []
    else:
        period_offsets = [80, 58, 33, 12]
        off_offsets = [74, 47, 21]

    for occurrence, offset in enumerate(period_offsets):
        active_date = (NOW - timedelta(days=offset + index % 4)).date()
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(
                    active_date,
                    time(_hour_for_time_period(time_period, occurrence), rng.randint(0, 55)),
                ),
            }
        )

    off_period = _different_time_period(time_period)
    for occurrence, offset in enumerate(off_offsets):
        active_date = (NOW - timedelta(days=offset + index % 4)).date()
        rows.append(
            {
                "user_id": user_id,
                "action_name": action,
                "event_time": datetime.combine(
                    active_date,
                    time(_hour_for_time_period(off_period, occurrence), rng.randint(0, 55)),
                ),
            }
        )

    truth = {
        "user_id": user_id,
        "action_name": action,
        "time_period": time_period,
        "negative_type": negative_type,
        "positive_label": 0,
        "pattern_id": f"time_period_negative_{index:03d}",
        "reason": "should_not_be_time_period_task",
    }
    return rows, truth


def _hour_for_time_period(time_period: str, occurrence: int) -> int:
    hours_by_period = {
        "early_morning": [1, 2, 4, 5],
        "morning": [8, 9, 10, 11],
        "afternoon": [13, 14, 16, 17],
        "evening": [19, 20, 21, 22],
    }
    hours = hours_by_period[time_period]
    return hours[occurrence % len(hours)]


def _different_time_period(time_period: str) -> str:
    periods = ["early_morning", "morning", "afternoon", "evening"]
    return periods[(periods.index(time_period) + 2) % len(periods)]


def _generate_sequence_events(
    user_id: str,
    sequence: list[str],
    user_index: int,
    sequence_index: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    repeat_count = 2 + (user_index + sequence_index) % 7
    rows: list[dict[str, object]] = []
    offsets = sorted(rng.sample(range(5, 80), repeat_count), reverse=True)
    for repeat, offset in enumerate(offsets):
        active_date = (NOW - timedelta(days=offset)).date()
        start = datetime.combine(
            active_date,
            time(8 + (user_index + repeat) % 12, rng.randint(0, 45)),
        )
        rows.extend(_session_rows(user_id, sequence, start, rng))

    truth = {
        "user_id": user_id,
        "sequence": " -> ".join(sequence),
        "sequence_length": len(sequence),
        "support_count": repeat_count,
        "next_action_candidate": sequence[-1],
        "positive_label": 1,
        "pattern_id": f"sequence_{user_index:03d}_{sequence_index}",
    }
    return rows, truth


def _session_rows(
    user_id: str,
    actions: Iterable[str],
    start_time: datetime,
    rng: random.Random,
) -> list[dict[str, object]]:
    current = start_time
    rows = []
    for action in actions:
        rows.append({"user_id": user_id, "action_name": action, "event_time": current})
        current += timedelta(minutes=rng.randint(1, 5))
    return rows


def _random_time_on(active_date: date, rng: random.Random) -> datetime:
    max_hour = 11 if active_date == NOW.date() else 21
    return datetime.combine(active_date, time(rng.randint(8, max_hour), rng.randint(0, 55)))


def _build_recent_ground_truth(events: pd.DataFrame) -> pd.DataFrame:
    window_start = NOW - timedelta(days=30)
    recent_events = events[
        (events["event_time"] > window_start) & (events["event_time"] <= NOW)
    ].copy()
    recent_events["event_date"] = pd.to_datetime(recent_events["event_time"]).dt.date
    stats = (
        recent_events.groupby("user_id")
        .agg(event_count_30d=("event_time", "size"), active_days_30d=("event_date", "nunique"))
        .reset_index()
    )
    all_users = pd.DataFrame({"user_id": sorted(events["user_id"].unique())})
    stats = all_users.merge(stats, on="user_id", how="left").fillna(0)
    stats["event_count_30d"] = stats["event_count_30d"].astype(int)
    stats["active_days_30d"] = stats["active_days_30d"].astype(int)
    active_event_counts = stats.loc[
        stats["event_count_30d"] > 0, "event_count_30d"
    ]
    p80 = active_event_counts.quantile(0.80) if not active_event_counts.empty else float("inf")

    def level(row: pd.Series) -> str:
        if row["event_count_30d"] == 0:
            return "low"
        if row["active_days_30d"] >= 15 or row["event_count_30d"] >= p80:
            return "high"
        if row["active_days_30d"] >= 4:
            return "middle"
        return "low"

    stats["expected_activity_level"] = stats.apply(level, axis=1)
    activity_by_user = stats.set_index("user_id")["expected_activity_level"].to_dict()
    rows: list[dict[str, object]] = []
    for item in stats.itertuples(index=False):
        selected = (
            events[(events["user_id"] == item.user_id) & (events["event_time"] <= NOW)]
            .sort_values(["event_time", "action_name"], ascending=[False, True])
            .head(3)
        )
        for rank, action_item in enumerate(selected.itertuples(index=False), start=1):
            rows.append(
                {
                    "user_id": item.user_id,
                    "action_name": action_item.action_name,
                    "event_time": action_item.event_time,
                    "expected_rank": rank,
                    "expected_activity_level": activity_by_user[item.user_id],
                }
            )
    return pd.DataFrame(rows).sort_values(["user_id", "expected_rank"]).reset_index(drop=True)


def _build_frequent_ground_truth(events: pd.DataFrame) -> pd.DataFrame:
    event_dates = events.assign(event_date=pd.to_datetime(events["event_time"]).dt.date)
    truth = (
        event_dates.groupby(["user_id", "action_name"])
        .agg(
            expected_event_count=("event_time", "size"),
            expected_active_days=("event_date", "nunique"),
            expected_last_time=("event_time", "max"),
        )
        .reset_index()
        .sort_values(
            [
                "user_id",
                "expected_event_count",
                "expected_active_days",
                "expected_last_time",
                "action_name",
            ],
            ascending=[True, False, False, False, True],
        )
    )
    truth["expected_rank"] = truth.groupby("user_id").cumcount() + 1
    truth = truth[truth["expected_rank"] <= 5]
    return truth[
        [
            "user_id",
            "action_name",
            "expected_event_count",
            "expected_active_days",
            "expected_rank",
        ]
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bank app portrait data.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="sample_data",
        help="Directory for sample_events and ground-truth CSV files.",
    )
    parser.add_argument("--profile", choices=["baseline", "large"], default="baseline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()

    paths = write_sample_dataset(
        args.output_dir,
        SampleDataConfig.for_profile(args.profile, seed=args.seed),
        encoding=args.encoding,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
