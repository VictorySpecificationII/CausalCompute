"""Command-line interface for CausalCompute."""
from __future__ import annotations

import argparse
import sys
from pprint import pprint

from .core.fundamentals import run_fundamentals
from .core.design import run_design
from .core.thermals import run_thermals
from .core.network import run_network
from .core.storage import run_storage
from .core.bom import run_bom
from .core.cost import run_cost
from .core.analysis import run_bottleneck
from .io.loader import load_brief
from .io.report import print_summary, print_story


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="causalcompute",
        description=(
            "Vendor-neutral, first-principles AI training infrastructure sizing.\n"
            "Supply a YAML brief; get compute, cluster, power, thermals, and network."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("brief", help="Path to a YAML brief file (e.g. briefs/13b.yaml)")
    p.add_argument(
        "--story",
        action="store_true",
        help="Print the Step-0 causal narrative and exit.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Dump full Step 0/1/2/3 dicts after the summary.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    try:
        brief = load_brief(args.brief)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error loading brief: {exc}", file=sys.stderr)
        sys.exit(1)

    # -- Step 0 ------------------------------------------------------------
    bundle0 = run_fundamentals(
        workload=brief.workload,
        state=brief.state,
        io=brief.io,
        device=brief.device,
        step=brief.step,
        schedule=brief.schedule,
        update=brief.update,
        fabric=brief.fabric,
        storage=brief.storage,
        checkpoint=brief.checkpoint,
    )

    if args.story:
        print_story(
            bundle0,
            workload_params=brief.workload.P,
            workload_tokens=brief.workload.Tok,
            deadline_s=brief.workload.T,
        )
        return

    # -- Step 1 ------------------------------------------------------------
    design = run_design(bundle0, G=brief.design_G, inputs=brief.design)

    # -- Bottleneck analysis -----------------------------------------------
    bottleneck_result = None
    if design["feasible"]:
        bottleneck_result = run_bottleneck(bundle0, design)

    # -- Step 2 ------------------------------------------------------------
    thermals_result = None
    if design["feasible"]:
        thermals_result = run_thermals(
            design,
            power=brief.power,
            thermals=brief.thermals,
            rack=brief.rack,
            T_run_s=brief.workload.T,
        )

    # -- Step 3 ------------------------------------------------------------
    network_result = None
    if design["feasible"]:
        network_result = run_network(design, network=brief.network)

    # -- Step 4 ------------------------------------------------------------
    storage_result = None
    if design["feasible"]:
        storage_result = run_storage(bundle0, design, storage=brief.storage_inputs)

    # -- Step 5 ------------------------------------------------------------
    bom_result = None
    if design["feasible"]:
        bom_result = run_bom(design, thermals_result, network_result, storage_result,
                             bundle0=bundle0, node_spec=brief.node_spec)

    # -- Step 6 ------------------------------------------------------------
    cost_result = None
    if bom_result is not None:
        cost_result = run_cost(bom_result, thermals_result,
                               bundle0=bundle0, cost=brief.cost)

    # -- Output ------------------------------------------------------------
    print_summary(bundle0, design, thermals_result, network_result, storage_result,
                  bom_result, cost_result, bottleneck_result)

    if args.debug:
        print("\n--- bundle0 (Step 0) ---")
        pprint(bundle0)
        print("\n--- design (Step 1) ---")
        pprint(design)
        print("\n--- thermals (Step 2) ---")
        pprint(thermals_result)
        print("\n--- network (Step 3) ---")
        pprint(network_result)
        print("\n--- storage (Step 4) ---")
        pprint(storage_result)
        print("\n--- bottleneck ---")
        pprint(bottleneck_result)
        print("\n--- bom (Step 5) ---")
        pprint(bom_result)
        print("\n--- cost (Step 6) ---")
        pprint(cost_result)


if __name__ == "__main__":
    main()
