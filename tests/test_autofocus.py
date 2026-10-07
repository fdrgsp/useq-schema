"""Tests for hardware and software autofocus plans and actions."""

import json

import pytest
import yaml
from pydantic import ValidationError

import useq
from useq import v2


def test_hardware_plan_as_action() -> None:
    plan = useq.AxesBasedAF(
        axes=("p",),
        autofocus_device_name="Z",
        autofocus_motor_offset=25.0,
        max_retries=5,
        search_below_um=20.0,
        search_above_um=10.0,
        search_step_um=2.5,
    )
    action = plan.as_action()
    assert isinstance(action, useq.HardwareAutofocus)
    assert action.type == "hardware_autofocus"
    assert action.autofocus_device_name == "Z"
    assert action.autofocus_motor_offset == 25.0
    assert action.max_retries == 5
    assert action.search_below_um == 20.0
    assert action.search_above_um == 10.0
    assert action.search_step_um == 2.5


def test_hardware_search_defaults_to_disabled() -> None:
    """Sequences built in code must not start moving Z on their own."""
    action = useq.AxesBasedAF(axes=("p",)).as_action()
    assert action.search_below_um == 0.0
    assert action.search_above_um == 0.0
    assert useq.HardwareAutofocus().search_below_um == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_below_um": -1.0},
        {"search_above_um": -1.0},
        {"search_step_um": -1.0},
        {"search_below_um": 10.0, "search_step_um": 0.0},
        {"search_above_um": 10.0, "search_step_um": 0.0},
    ],
)
def test_invalid_search_settings(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        useq.HardwareAutofocus(**kwargs)


def test_software_plan_as_action() -> None:
    plan = useq.SoftwareAxesBasedAF(
        axes=("p", "t"),
        method="oughtafocus",
        focus_device="ZPiezo",
        settings={"search_range_um": 15.0, "scoring": "Edges"},
        max_retries=2,
    )
    action = plan.as_action()
    assert isinstance(action, useq.SoftwareAutofocus)
    assert action.type == "software_autofocus"
    assert action.method == "oughtafocus"
    assert action.focus_device == "ZPiezo"
    assert action.settings == {"search_range_um": 15.0, "scoring": "Edges"}
    assert action.max_retries == 2


def test_software_settings_must_be_serializable() -> None:
    with pytest.raises(ValidationError, match="must be JSON serializable"):
        useq.SoftwareAutofocus(method="oughtafocus", settings={"bad": object()})
    with pytest.raises(ValidationError, match="must be JSON serializable"):
        useq.SoftwareAxesBasedAF(
            axes=("p",), method="oughtafocus", settings={"bad": object()}
        )


# -------------------------- plan discrimination ---------------------------


def test_plan_discrimination() -> None:
    """A plan that names a `method` is a software plan; otherwise hardware."""
    soft = useq.MDASequence(
        autofocus_plan={"axes": ("p",), "method": "jaf_hp", "settings": {"a": 1}}
    )
    assert isinstance(soft.autofocus_plan, useq.SoftwareAxesBasedAF)
    assert soft.autofocus_plan.method == "jaf_hp"
    assert soft.autofocus_plan.settings == {"a": 1}

    hard = useq.MDASequence(
        autofocus_plan={"axes": ("p",), "autofocus_motor_offset": 10.0}
    )
    assert isinstance(hard.autofocus_plan, useq.AxesBasedAF)
    assert not isinstance(hard.autofocus_plan, useq.SoftwareAxesBasedAF)


def test_legacy_hardware_plan_still_parses() -> None:
    """JSON written before software autofocus existed must keep its meaning."""
    legacy = '{"autofocus_plan": {"autofocus_device_name": "Z", "axes": ["c"]}}'
    seq = useq.MDASequence.model_validate_json(legacy)
    assert isinstance(seq.autofocus_plan, useq.AxesBasedAF)
    assert seq.autofocus_plan.autofocus_device_name == "Z"
    assert seq.autofocus_plan.axes == ("c",)


@pytest.mark.parametrize(
    "plan",
    [
        useq.AxesBasedAF(axes=("p",), autofocus_motor_offset=10.0, search_above_um=8.0),
        useq.SoftwareAxesBasedAF(
            axes=("p",), method="oughtafocus", settings={"crop_factor": 0.5}
        ),
    ],
    ids=["hardware", "software"],
)
def test_plan_round_trip(plan: useq.AnyAutofocusPlan) -> None:
    seq = useq.MDASequence(autofocus_plan=plan)
    rebuilt = [
        useq.MDASequence.model_validate_json(seq.model_dump_json()),
        useq.MDASequence.model_validate(yaml.safe_load(seq.yaml())),
    ]
    for other in rebuilt:
        assert other.autofocus_plan == plan
        assert type(other.autofocus_plan) is type(plan)
    # the action a plan builds must survive a round trip of its own
    action = plan.as_action()
    dumped = action.model_dump_json()
    assert json.loads(dumped)["type"] == action.type
    assert type(action).model_validate_json(dumped) == action


# ----------------------------- event insertion -----------------------------


def _af_events(seq: useq.MDASequence | v2.MDASequence) -> list[useq.MDAEvent]:
    not_af = (useq.AcquireImage, useq.CustomAction)
    return [e for e in seq if not isinstance(e.action, not_af)]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_software_autofocus_events(version: str) -> None:
    cls = useq.MDASequence if version == "v1" else v2.MDASequence
    seq = cls(
        stage_positions=[(0, 0, 0), (1, 1, 1)],
        z_plan={"range": 2.0, "step": 1.0},
        autofocus_plan={"axes": ("p",), "method": "oughtafocus"},
    )
    af = _af_events(seq)
    # one autofocus event per position, and none at the other z slices
    assert len(af) == 2
    assert [e.index["p"] for e in af] == [0, 1]
    for event in af:
        assert isinstance(event.action, useq.SoftwareAutofocus)
        assert event.action.method == "oughtafocus"


def test_software_autofocus_uses_home_z_position() -> None:
    """Autofocus runs at the middle of a relative z stack, not its first slice."""
    seq = useq.MDASequence(
        stage_positions=[(0, 0, 0)],
        z_plan={"range": 2.0, "step": 1.0},
        autofocus_plan={"axes": ("p",), "method": "oughtafocus"},
    )
    events = list(seq)
    assert events[0].z_pos == 0.0  # the autofocus event: the home position
    assert events[1].z_pos == -1.0  # the first slice of the stack


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("every_n", [1, 2, 3])
def test_every_n_timepoints(version: str, every_n: int) -> None:
    cls = useq.MDASequence if version == "v1" else v2.MDASequence
    seq = cls(
        time_plan={"interval": 0, "loops": 6},
        autofocus_plan={"axes": ("t",), "every_n_timepoints": every_n},
    )
    assert [e.index["t"] for e in _af_events(seq)] == list(range(0, 6, every_n))


def test_every_n_timepoints_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        useq.AxesBasedAF(axes=("t",), every_n_timepoints=0)


def test_no_autofocus_without_matching_axis() -> None:
    seq = useq.MDASequence(
        time_plan={"interval": 0, "loops": 3},
        autofocus_plan={"axes": ("p",), "method": "oughtafocus"},
    )
    assert _af_events(seq) == []


# --------------------------- absolute z rejection ---------------------------


@pytest.mark.parametrize(
    "plan",
    [
        {"axes": ("p",)},
        {"axes": ("p",), "method": "oughtafocus"},
    ],
    ids=["hardware", "software"],
)
def test_absolute_z_plan_rejected(plan: dict) -> None:
    with pytest.raises(ValidationError, match="Absolute Z positions"):
        useq.MDASequence(
            z_plan={"top": 2.0, "bottom": 0.0, "step": 1.0},
            autofocus_plan=plan,
        )


@pytest.mark.parametrize(
    "plan",
    [
        {"axes": ("p",)},
        {"axes": ("p",), "method": "oughtafocus"},
    ],
    ids=["hardware", "software"],
)
def test_absolute_z_plan_rejected_per_position(plan: dict) -> None:
    with pytest.raises(ValidationError, match="Absolute Z positions"):
        useq.MDASequence(
            z_plan={"top": 2.0, "bottom": 0.0, "step": 1.0},
            stage_positions=[
                {"x": 0, "y": 0, "sequence": {"autofocus_plan": plan}},
            ],
        )
