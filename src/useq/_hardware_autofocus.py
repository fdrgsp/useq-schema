"""Plans for autofocus performed by a hardware autofocus device."""

from pydantic import Field

from useq._actions import HardwareAutofocus
from useq._autofocus_base import _AutofocusPlanBase, _AxesTrigger


class AutoFocusPlan(_AutofocusPlanBase):
    """Base class for hardware autofocus plans.

    Attributes
    ----------
    autofocus_device_name : str | None
        Optional name of the offset motor device.  If `None`, acquisition engines may
        attempt to set the offset however they see fit (such as using a current
        or default autofocus device.)
    autofocus_motor_offset : float | None
        Before autofocus is performed, the autofocus motor should be moved to this
        offset, if applicable. (Not all autofocus devices have an offset motor.)
        If None, the autofocus motor should not be moved.
    max_retries : int
        The number of retries if autofocus fails. By default, 3.
    search_below_um : float
        If autofocus fails at the current focus position, acquisition engines may step
        the focus device *down* by `search_step_um` at a time, up to this distance in
        µm, retrying autofocus at each step.  By default, `0.0`: no search.
    search_above_um : float
        As `search_below_um`, but stepping *up*.  By default, `0.0`.
    search_step_um : float
        Step size in µm used while searching.  By default, `5.0`.
    """

    autofocus_device_name: str | None = None
    autofocus_motor_offset: float | None = None
    max_retries: int = 3
    search_below_um: float = Field(default=0.0, ge=0.0)
    search_above_um: float = Field(default=0.0, ge=0.0)
    search_step_um: float = Field(default=5.0, ge=0.0)

    def as_action(self) -> HardwareAutofocus:
        """Return a [`useq.HardwareAutofocus`][] for this autofocus plan."""
        return HardwareAutofocus(
            autofocus_device_name=self.autofocus_device_name,
            autofocus_motor_offset=self.autofocus_motor_offset,
            max_retries=self.max_retries,
            search_below_um=self.search_below_um,
            search_above_um=self.search_above_um,
            search_step_um=self.search_step_um,
        )


class AxesBasedAF(_AxesTrigger, AutoFocusPlan):
    """Hardware autofocus plan that fires when any of the specified axes change.

    Attributes
    ----------
    axes : Tuple[str, ...]
        Tuple of axis label to use for hardware autofocus.  At every event in which
        *any* axis in this tuple is change, autofocus will be performed.  For example,
        if `axes` is `('p',)` then autofocus will be performed every time the `p` axis
        is change, (in other words: every time the position is changed.).
    every_n_timepoints : int
        Only autofocus on time points whose index is a multiple of this number.
        By default, `1`: every time point.
    """
