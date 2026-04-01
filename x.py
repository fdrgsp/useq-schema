"""Test MDASequence iteration order with different axis_order combinations.

Tests whether axis_order correctly dictates iteration order, especially
when a position has a sub-sequence with grid_plan.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import useq


def axis_order_from_events(events: list) -> str:
    """Derive the effective axis order from the event index sequence.

    The effective axis order is determined by which axis changes slowest
    (outermost) to fastest (innermost). We do this by counting the number
    of contiguous "runs" for each axis -- a higher run count means the axis
    changes more frequently (more inner).
    """
    if not events:
        return ""
    all_keys = list(events[0].index.keys())
    if not all_keys:
        return ""

    run_counts = {}
    for key in all_keys:
        vals = [e.index.get(key, 0) for e in events]
        runs = 1 + sum(1 for i in range(1, len(vals)) if vals[i] != vals[i - 1])
        run_counts[key] = runs

    # sort by run count ascending: slowest (outermost) first
    sorted_axes = sorted(run_counts, key=lambda k: run_counts[k])
    return "".join(sorted_axes)


def print_events(events: list, label: str = "") -> None:
    print(f"\n{label}")
    print(f"  total events: {len(events)}")
    if not events:
        return
    effective = axis_order_from_events(events)
    print(f"  effective order (slowest→fastest): {effective}")
    # print first few and last few
    for e in events[:4]:
        print(f"    {dict(e.index)}")
    if len(events) > 8:
        print(f"    ... ({len(events) - 8} more)")
    for e in events[-4:]:
        print(f"    {dict(e.index)}")


def make_seq_with_subgrid(axis_order: str) -> useq.MDASequence:
    """Sequence where position has sub-sequence with grid_plan."""
    return useq.MDASequence(
        axis_order=axis_order,
        channels=["DAPI", "FITC"],
        time_plan={"interval": 0, "loops": 2},
        z_plan={"range": 2, "step": 1},  # 3 z steps: z=0,1,2
        stage_positions=[
            useq.Position(
                x=1,
                y=2,
                z=3,
                name="pos1",
                sequence=useq.MDASequence(
                    grid_plan=useq.GridRowsColumns(rows=2, columns=2)
                ),
            )
        ],
    )


def make_seq_with_global_grid(axis_order: str) -> useq.MDASequence:
    """Sequence with global grid_plan on the parent (no sub-sequence)."""
    return useq.MDASequence(
        axis_order=axis_order,
        channels=["DAPI", "FITC"],
        time_plan={"interval": 0, "loops": 2},
        z_plan={"range": 2, "step": 1},  # 3 z steps
        grid_plan=useq.GridRowsColumns(rows=2, columns=2),
        stage_positions=[
            useq.Position(x=1, y=2, z=3, name="pos1"),
        ],
    )


# ─── Reference: global grid_plan (known to work) ───────────────────────────
print("=" * 70)
print("REFERENCE: global grid_plan on parent (no sub-sequence)")
print("Expected: axis_order is fully respected for all axes including g")
print("=" * 70)

for order in ["tpgcz", "tpgzc", "tpcgz", "tpczg", "tpzcg", "tpzgc"]:
    try:
        seq = make_seq_with_global_grid(order)
        events = list(seq)
        print_events(events, f"axis_order={order!r}")
    except Exception as ex:
        print(f"\naxis_order={order!r}: ERROR — {ex}")

# ─── Sub-sequence grid_plan: axes t, p, c, z (no g in parent) ──────────────
print("\n" + "=" * 70)
print("SUB-SEQUENCE grid_plan — testing axis_order for t, p, c, z")
print("(g comes from sub-sequence, always ends up innermost)")
print("=" * 70)

for order in ["tpczg", "tpcgz", "tpgcz", "tpzcg", "tpzgc", "tpgzc",
              "tpcz", "tpzc", "ptcz", "tcpz"]:
    try:
        seq = make_seq_with_subgrid(order)
        events = list(seq)
        print_events(events, f"axis_order={order!r}")
    except Exception as ex:
        print(f"\naxis_order={order!r}: ERROR — {ex}")

# ─── Focus: does axis_order of t,c,z (non-g, non-p) get respected? ─────────
print("\n" * 1 + "=" * 70)
print("FOCUS: verifying t/c/z relative order with sub-sequence grid")
print("Each pair shows the axis that changes fastest (innermost among t,c,z)")
print("=" * 70)

# The three axes we can reorder freely: t, c, z (p is fixed = 1 position)
for order in ["tpcz", "tpzc", "ptcz", "ptzc", "pctz", "pczt",
              "pzct", "pztc", "cptz", "czpt"]:
    try:
        seq = make_seq_with_subgrid(order)
        events = list(seq)
        effective = axis_order_from_events(events)
        # Axes with only 1 distinct value can't reveal ordering information;
        # exclude them from both sides before comparing.
        single_value = {
            k for k in (events[0].index if events else {})
            if len({e.index.get(k) for e in events}) <= 1
        }
        def _strip(s: str, _skip: frozenset = frozenset(single_value | {"g"})) -> str:
            return "".join(c for c in s if c not in _skip)
        match = _strip(effective) == _strip(order)
        print(f"  declared={order!r:12s} effective={effective!r:12s}  "
              f"match={'YES' if match else 'NO ←'}")
    except Exception as ex:
        print(f"  declared={order!r}: ERROR — {ex}")
