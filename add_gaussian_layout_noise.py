#!/usr/bin/env python3
"""Add independent Gaussian noise to layout placements and write a new layout."""

import argparse
import json
from pathlib import Path

import numpy as np

from geometry import validate_layout


HERE = Path(__file__).resolve().parent


def main(args: argparse.Namespace | None = None) -> dict[str, dict]:

    data = json.loads(args.input.read_text())
    if not isinstance(data, dict) or not data:
        parser.error("input must be a non-empty JSON object of named layouts")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")

    spec = json.loads((HERE / "challenge_spec.json").read_text())
    rng = np.random.default_rng(args.seed)
    for name, layout in data.items():
        if not isinstance(layout, dict) or not isinstance(layout.get("placements_mm"), dict):
            parser.error(f"layout '{name}' must contain an object-valued placements_mm")
        source_placements = layout["placements_mm"]
        for die, coords in source_placements.items():
            if (not isinstance(coords, list) or len(coords) != 2
                    or any(isinstance(coord, bool) or not isinstance(coord, (int, float))
                           for coord in coords)):
                parser.error(f"placements_mm for '{name}/{die}' must be a pair of numbers")
        for attempt in range(1, args.max_attempts + 1):
            placements = {
                die: [round(float(coord + delta), 6)
                      for coord, delta in zip(coords, rng.normal(0.0, 1.0, size=2))]
                for die, coords in source_placements.items()
            }
            try:
                routing = validate_layout(spec, placements)
            except ValueError:
                continue
            layout["placements_mm"] = placements
            layout["attempt"] = attempt
            layout["weighted_route_gb_s_mm"] = routing["weighted_route_gb_s_mm"]
            break
        else:
            parser.error(f"no valid Gaussian-perturbed layout found for '{name}' "
                         f"after {args.max_attempts} attempts")
        layout["random_seed"] = args.seed
        layout["sampling"] = (
            "independent additive Normal(0, 1) mm noise per placement coordinate; "
            "reject samples that violate layout constraints"
        )

    print(f"Wrote Gaussian-perturbed layouts to {args.output}")
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "random_layout_v1.json")
    parser.add_argument("--output", type=Path, default=HERE / "random_layout.json")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--max-attempts", type=int, default=10000)
    args = parser.parse_args()
    data = main(args)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
