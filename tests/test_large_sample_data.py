from __future__ import annotations

from pathlib import Path

from user_portrait.evaluation import evaluate_profile_outputs
from user_portrait.large_sample_data import build_large_dataset_manifest
from user_portrait.sample_data import SampleDataConfig, generate_sample_dataset


def test_large_profile_has_expected_scale_and_scenario_coverage() -> None:
    config = SampleDataConfig.for_profile("large", seed=20260701)
    frames = generate_sample_dataset(config)
    events = frames["sample_events"]
    periodic = frames["sample_ground_truth_periodic"]
    time_injected = frames["sample_injected_time_period_patterns"]
    sequences = frames["sample_ground_truth_sequences"]
    manifest = build_large_dataset_manifest(frames, config)

    assert events["user_id"].nunique() == 2000
    assert 140_000 <= len(events) <= 220_000
    assert len(frames["sample_events_dirty"]) == len(events) + 100
    assert periodic["period_type"].value_counts().to_dict() == {
        "daily": 250,
        "weekly": 250,
        "monthly": 250,
        "interval": 250,
    }
    assert time_injected["time_period"].value_counts().to_dict() == {
        "early_morning": 250,
        "morning": 250,
        "afternoon": 250,
        "evening": 250,
    }
    assert sequences["sequence_length"].value_counts().to_dict() == {
        2: 300,
        3: 300,
        4: 300,
        5: 300,
    }
    injected_actions = set(time_injected["action_name"])
    injected_hours = set(
        events.loc[events["action_name"].isin(injected_actions), "event_time"].dt.hour
    )
    assert {0, 5, 6, 11, 12, 17, 18, 23}.issubset(injected_hours)
    assert manifest["time_period_ground_truth_scope"] == "exhaustive"
    assert manifest["overlap_users"]["at_least_two_modules"] >= 400
    assert manifest["overlap_users"]["at_least_three_modules"] >= 150


def test_large_generated_outputs_pass_exhaustive_and_dirty_evaluation() -> None:
    root = Path(__file__).parents[1]

    report = evaluate_profile_outputs(
        root / "sample_data_large",
        root / "portrait_output_large",
        dirty_profiles_dir=root / "portrait_output_large_dirty",
    )

    time_period = report["modules"]["time_period_tasks"]
    assert time_period["ground_truth_scope"] == "exhaustive"
    assert time_period["metrics"]["precision"]["value"] == 1.0
    assert time_period["metrics"]["recall"]["value"] == 1.0
    assert time_period["metrics"]["rank_accuracy"]["value"] == 1.0
    assert (
        report["modules"]["dirty_input_consistency"]["metrics"]["table_match_rate"][
            "value"
        ]
        == 1.0
    )
    assert report["overall_pass"] is True
