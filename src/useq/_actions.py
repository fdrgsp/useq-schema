from typing import Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator
from pydantic_core import PydanticSerializationError

from useq._base_model import FrozenModel

_dict_adapter = TypeAdapter(dict, config=ConfigDict(defer_build=True))


def _ensure_json_serializable(data: dict, name: str) -> dict:
    """Raise `ValueError` unless `data` can be serialized to JSON by pydantic."""
    try:
        _dict_adapter.serializer.to_json(data)
    except PydanticSerializationError as e:
        raise ValueError(
            f"`{name}` must be JSON serializable, but is not:\n  {e}.\n"
            "  (You may use a pydantic object for custom serialization).\n   "
        ) from e
    return data


class Action(FrozenModel):
    """Base class for a [`useq.MDAEvent`][] action.

    An `Action` specifies what task should be performed during a
    [`useq.MDAEvent`][]. An `Action` can be for example used to acquire an
    image ([`useq.AcquireImage`][]) or to perform a hardware autofocus
    ([`useq.HardwareAutofocus`][]).  An action of `None` implies `AcquireImage`.

    You may use `CustomAction` to indicate any custom action, with the `data` attribute
    containing any data required to perform the custom action.

    Attributes
    ----------
    type : str
        Type of the action that should be performed at the [`useq.MDAEvent`][].
    """

    type: str


class AcquireImage(Action):
    """[`useq.Action`][] to acquire an image.

    Attributes
    ----------
    type : Literal["acquire_image"]
        This action can be used to acquire an image.
    """

    type: Literal["acquire_image"] = "acquire_image"  # pyright: ignore[reportIncompatibleVariableOverride]


class HardwareAutofocus(Action):
    """[`useq.Action`][] to perform a hardware autofocus.

    See also [`useq.AutoFocusPlan`][].

    Attributes
    ----------
    type : Literal["hardware_autofocus"]
        This action can be used to trigger hardware autofocus.
    autofocus_device_name : str, optional
        The name of the autofocus offset motor device (if applicable).  If `None`,
        acquisition engines may attempt to set the offset however they see fit (such as
        using a current or default autofocus device.)
    autofocus_motor_offset: float, optional
        Before autofocus is performed, the autofocus motor should be moved to this
        offset, if applicable. (Not all autofocus devices have an offset motor.)
        If None, the autofocus motor should not be moved.
    max_retries : int
        The number of retries if autofocus fails. By default, 3.
    search_below_um : float
        If autofocus fails at the current focus position, acquisition engines may step
        the focus device *down* by `search_step_um` at a time, up to this distance in
        µm, retrying autofocus at each step.  Hardware autofocus devices can only lock
        within a limited range, so a large move (a new well, a tilted sample, drift) can
        leave the sample outside of it.  By default, `0.0`: no search is performed and
        autofocus is only attempted at the current position.
    search_above_um : float
        As `search_below_um`, but stepping *up*.  Engines should search below before
        searching above.  By default, `0.0`.
    search_step_um : float
        Step size in µm used while searching for a position at which autofocus succeeds.
        Only meaningful if `search_below_um` or `search_above_um` is greater than zero.
        By default, `5.0`.
    """

    type: Literal["hardware_autofocus"] = "hardware_autofocus"  # pyright: ignore[reportIncompatibleVariableOverride]
    autofocus_device_name: str | None = None
    autofocus_motor_offset: float | None = None
    max_retries: int = 3
    search_below_um: float = Field(default=0.0, ge=0.0)
    search_above_um: float = Field(default=0.0, ge=0.0)
    search_step_um: float = Field(default=5.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_search(self) -> "HardwareAutofocus":
        if (self.search_below_um or self.search_above_um) and not self.search_step_um:
            raise ValueError(
                "`search_step_um` must be greater than zero when `search_below_um` or "
                "`search_above_um` is used."
            )
        return self


class SoftwareAutofocus(Action):
    """[`useq.Action`][] to perform an image-based (software) autofocus.

    This action names a routine and carries its settings, but does not define which
    routines exist or what their settings mean: `method` and `settings` are interpreted
    by the acquisition engine.  This keeps `useq` free of any particular engine's
    catalog of algorithms, and lets a saved sequence travel with the exact settings it
    was run with.

    See also [`useq.SoftwareAxesBasedAF`][].

    Attributes
    ----------
    type : Literal["software_autofocus"]
        This action can be used to trigger a software autofocus.
    method : str
        Name of the autofocus routine, as known to the acquisition engine.
    focus_device : str, optional
        Name of the stage device the routine should move.  If `None`, acquisition
        engines should use their default focus device.
    settings : dict, optional
        Routine-specific settings. It *must* be serializable to JSON by `pydantic`.
    max_retries : int
        The number of attempts if the routine fails. By default, 1 (no retry).
    """

    type: Literal["software_autofocus"] = "software_autofocus"  # pyright: ignore[reportIncompatibleVariableOverride]
    method: str
    focus_device: str | None = None
    settings: dict = Field(default_factory=dict)
    max_retries: int = Field(default=1, ge=1)

    @field_validator("settings", mode="after")
    @classmethod
    def _ensure_serializable(cls, settings: dict) -> dict:
        return _ensure_json_serializable(settings, "SoftwareAutofocus.settings")


class CustomAction(Action):
    """[`useq.Action`][] to perform a custom action.

    This is a generic user action that can be used to represent anything that is not
    covered by the other action types, such as a microfluidic event, or a
    photostimulation, etc...

    The `data` attribute is a dictionary that can contain any data that is needed to
    perform the custom action.  It *must* be serializable to JSON by `pydantic`.

    Attributes
    ----------
    type : Literal["custom"]
        This action can be used to perform a custom action.
    name : str, optional
        A name for the custom action (not to be confused with the `type` attribute,
        which must always be `"custom"`).
    data : dict, optional
        Custom data associated with the action.
    """

    type: Literal["custom"] = "custom"  # pyright: ignore[reportIncompatibleVariableOverride]
    name: str = ""
    data: dict = Field(default_factory=dict)

    @field_validator("data", mode="after")
    @classmethod
    def _ensure_serializable(cls, data: dict) -> dict:
        return _ensure_json_serializable(data, "CustomAction.data")


AnyAction = HardwareAutofocus | SoftwareAutofocus | AcquireImage | CustomAction
