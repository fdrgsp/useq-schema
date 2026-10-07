"""Plans for image-based (software) autofocus."""

from pydantic import Field, field_validator

from useq._actions import SoftwareAutofocus, _ensure_json_serializable
from useq._autofocus_base import _AutofocusPlanBase, _AxesTrigger


class SoftwareAutofocusPlan(_AutofocusPlanBase):
    """Base class for software (image-based) autofocus plans.

    `useq` does not define which routines exist: `method` and `settings` are
    interpreted by the acquisition engine.

    Attributes
    ----------
    method : str
        Name of the autofocus routine, as known to the acquisition engine.
    focus_device : str | None
        Name of the stage device the routine should move.  If `None`, acquisition
        engines should use their default focus device.
    settings : dict
        Routine-specific settings. It *must* be serializable to JSON by `pydantic`.
    max_retries : int
        The number of attempts if the routine fails. By default, 1 (no retry).
    """

    method: str
    focus_device: str | None = None
    settings: dict = Field(default_factory=dict)
    max_retries: int = Field(default=1, ge=1)

    @field_validator("settings", mode="after")
    @classmethod
    def _ensure_serializable(cls, settings: dict) -> dict:
        return _ensure_json_serializable(settings, f"{cls.__name__}.settings")

    def as_action(self) -> SoftwareAutofocus:
        """Return a [`useq.SoftwareAutofocus`][] for this autofocus plan."""
        return SoftwareAutofocus(
            method=self.method,
            focus_device=self.focus_device,
            settings=self.settings,
            max_retries=self.max_retries,
        )


class SoftwareAxesBasedAF(_AxesTrigger, SoftwareAutofocusPlan):
    """Software autofocus plan that fires when any of the specified axes change.

    Attributes
    ----------
    axes : Tuple[str, ...]
        Tuple of axis label to use for software autofocus.  At every event in which
        *any* axis in this tuple is change, autofocus will be performed.
    every_n_timepoints : int
        Only autofocus on time points whose index is a multiple of this number.
        By default, `1`: every time point.
    method : str
        Name of the autofocus routine, as known to the acquisition engine.
    focus_device : str | None
        Name of the stage device the routine should move.
    settings : dict
        Routine-specific settings.
    max_retries : int
        The number of attempts if the routine fails. By default, 1 (no retry).
    """
