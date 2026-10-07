"""Shared base classes for autofocus plans.

Concrete plans live in `useq._hardware_autofocus` and `useq._software_autofocus`;
the union of them is `useq._autofocus.AnyAutofocusPlan`.
"""

from typing import Any

from pydantic import Field, PrivateAttr

from useq._actions import HardwareAutofocus, SoftwareAutofocus
from useq._base_model import FrozenModel
from useq._enums import Axis
from useq._mda_event import MDAEvent


class _AutofocusPlanBase(FrozenModel):
    """Behaviour shared by every autofocus plan.

    A plan decides *when* autofocus should happen (`should_autofocus`) and *what*
    should happen (`as_action`).  Subclasses must implement both.
    """

    def as_action(self) -> HardwareAutofocus | SoftwareAutofocus:
        """Return the [`useq.Action`][] that performs autofocus for this plan."""
        raise NotImplementedError("as_action() must be implemented by subclass.")

    def event(self, event: MDAEvent) -> MDAEvent | None:
        """Return an autofocus [`useq.MDAEvent`][] if autofocus should be performed.

        The z position of the new [`useq.MDAEvent`][] is also updated if a relative
        zplan is provided since autofocus shuld be performed on the home z stack
        position.
        """
        if not self.should_autofocus(event):
            return None

        updates: dict[str, Any] = {"action": self.as_action()}
        if event.z_pos is not None and event.sequence is not None:
            zplan = event.sequence.z_plan
            if zplan and zplan.is_relative and "z" in event.index:
                updates["z_pos"] = event.z_pos - list(zplan)[event.index["z"]]

        return event.model_copy(update=updates)

    def should_autofocus(self, event: MDAEvent) -> bool:
        """Method that must be implemented by a subclass.

        Should return True if autofocus should be performed (see
        [`useq.AxesBasedAF`][]).
        """
        raise NotImplementedError("should_autofocus() must be implemented by subclass.")


class _AxesTrigger(FrozenModel):
    """Trigger autofocus whenever any of `axes` changes.

    Shared by [`useq.AxesBasedAF`][] and [`useq.SoftwareAxesBasedAF`][].

    Attributes
    ----------
    axes : Tuple[str, ...]
        Tuple of axis label to use for autofocus.  At every event in which
        *any* axis in this tuple is change, autofocus will be performed.  For example,
        if `axes` is `('p',)` then autofocus will be performed every time the `p` axis
        is change, (in other words: every time the position is changed.).
    every_n_timepoints : int
        Only autofocus on time points whose index is a multiple of this number,
        regardless of which axis triggered it.  By default, `1`: every time point.
    """

    axes: tuple[str, ...]
    every_n_timepoints: int = Field(default=1, ge=1)
    _previous: dict = PrivateAttr(default_factory=dict)

    def should_autofocus(self, event: MDAEvent) -> bool:
        """Return `True` if autofocus should be performed at this event.

        Will return `True` if any of the axes specified in `axes` have changed from the
        previous event, unless `every_n_timepoints` excludes this time point.
        """
        self._previous, previous = dict(event.index), self._previous
        if not any(
            axis in self.axes and previous.get(axis) != index
            for axis, index in event.index.items()
        ):
            return False
        return self.allows_timepoint(event.index.get(Axis.TIME))

    def allows_timepoint(self, t_index: int | None) -> bool:
        """Return `True` unless `every_n_timepoints` excludes time point `t_index`."""
        if t_index is None or self.every_n_timepoints <= 1:
            return True
        return t_index % self.every_n_timepoints == 0
