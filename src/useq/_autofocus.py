"""The union of all autofocus plans, and how they are told apart."""

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Discriminator, Tag

from useq._hardware_autofocus import AutoFocusPlan, AxesBasedAF
from useq._software_autofocus import SoftwareAutofocusPlan, SoftwareAxesBasedAF

__all__ = [
    "AnyAutofocusPlan",
    "AutoFocusPlan",
    "AxesBasedAF",
    "SoftwareAutofocusPlan",
    "SoftwareAxesBasedAF",
]


def _autofocus_plan_tag(value: Any) -> str:
    """Pick the plan variant for `AnyAutofocusPlan`.

    `FrozenModel` ignores extra fields, so a plain union could parse a software plan as
    a hardware one and silently drop `method`.  A software plan is the one that names a
    `method`.
    """
    if isinstance(value, Mapping):
        return "software" if "method" in value else "hardware"
    return "software" if isinstance(value, SoftwareAutofocusPlan) else "hardware"


AnyAutofocusPlan = Annotated[
    Annotated[AxesBasedAF, Tag("hardware")]
    | Annotated[SoftwareAxesBasedAF, Tag("software")],
    Discriminator(_autofocus_plan_tag),
]
