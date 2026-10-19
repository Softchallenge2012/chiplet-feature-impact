#!/usr/bin/env python3
"""Generate synthetic layouts and label each by nearest reference layout from data/*.json."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import add_gaussian_layout_noise as noise_gen
from train_knn_from_layout import DATA_DIR, LABELS_FILE, extract_layout_vector, load_labels


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "random_layout.json"
DEFAULT_OUTPUT = DATA_DIR / "synthetic_similarity_labeled_1000.csv"


class _ParserShim:
    """Provide parser.error used by add_gaussian_layout_noise.main."""

    @staticmethod
    def error(message: str) -> None:
        raise ValueError(message)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_layout_records(payload: Any) -> list[tuple[dict[str, Any], int]]:
    records: list[tuple[dict[str, Any], int]] = []

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, dict) and "layout_config" in item:
                records.append((item, idx))
        return records

    if isinstance(payload, dict):
        if "layout_config" in payload:
            records.append((payload, 0))
            return records

        if "records" in payload and isinstance(payload["records"], list):
            for idx, item in enumerate(payload["records"]):
                if isinstance(item, dict) and "layout_config" in item:
                    records.append((item, idx))
            return records

    return records


def _resolve_label(record: dict[str, Any], filename: str, labels_map: dict[str, int]) -> int | None:
    for key in ("predicted_label", "label", "class"):
        if key in record:
            value = record[key]
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                return int(value.strip())

    return labels_map.get(filename)


def build_reference_set(data_dir: Path, labels_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []

    for json_path in sorted(data_dir.glob("*.json")):
        payload = _read_json(json_path)
        records = _iter_layout_records(payload)
        for record, idx in records:
            label = _resolve_label(record, json_path.name, labels_map)
            if label is None:
                continue
            vector = extract_layout_vector(record["layout_config"])
            vectors.append(vector)
            labels.append(label)
            meta.append({"filename": json_path.name, "record_index": idx, "label": label})

    if not vectors:
        raise ValueError(f"No labeled layout_config records found in {data_dir}")

    return np.vstack(vectors), np.asarray(labels, dtype=np.int64), meta


def _generate_one(base_input_path: Path, seed: int, max_attempts: int) -> dict[str, Any]:
    args = argparse.Namespace(
        input=base_input_path,
        output=Path("/dev/null"),
        seed=seed,
        max_attempts=max_attempts,
    )
    generated = noise_gen.main(args)
    first_layout_name = next(iter(generated.keys()))
    return generated[first_layout_name]


def write_csv(rows: list[dict[str, Any]], output_path: Path, die_keys: list[str]) -> None:
    fieldnames = [
        "sample_id",
        "seed",
        "predicted_label",
        "nearest_source_file",
        "nearest_record_index",
        "distance",
    ]
    for die in die_keys:
        fieldnames.append(f"{die}_x")
        fieldnames.append(f"{die}_y")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=20261018)
    parser.add_argument("--max-attempts", type=int, default=10000)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    noise_gen.parser = _ParserShim()

    labels_map = {filename: label for filename, label in load_labels(LABELS_FILE)}
    ref_x, ref_y, ref_meta = build_reference_set(DATA_DIR, labels_map)

    scaler = StandardScaler()
    ref_x_scaled = scaler.fit_transform(ref_x)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ref_x_scaled)

    base_data = _read_json(args.input)
    if not isinstance(base_data, dict) or not base_data:
        raise ValueError("Input JSON must be a non-empty object of named layouts")

    records_out: list[dict[str, Any]] = []
    die_keys: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_input_path = Path(tmp_dir) / "base_layout.json"
        tmp_input_path.write_text(json.dumps(base_data), encoding="utf-8")

        for i in range(args.num_samples):
            seed = args.seed_start + i
            generated_layout = _generate_one(tmp_input_path, seed=seed, max_attempts=args.max_attempts)
            layout_config = generated_layout["placements_mm"]

            if not die_keys:
                die_keys = sorted(layout_config.keys())

            vector = extract_layout_vector(layout_config).reshape(1, -1)
            vector_scaled = scaler.transform(vector)
            distances, indices = nn.kneighbors(vector_scaled, n_neighbors=1, return_distance=True)

            nearest_idx = int(indices[0][0])
            nearest_meta = ref_meta[nearest_idx]
            predicted_label = int(ref_y[nearest_idx])
            distance = float(distances[0][0])

            row: dict[str, Any] = {
                "sample_id": i + 1,
                "seed": seed,
                "predicted_label": predicted_label,
                "nearest_source_file": nearest_meta["filename"],
                "nearest_record_index": int(nearest_meta["record_index"]),
                "distance": distance,
            }

            for die in die_keys:
                x, y = layout_config[die]
                row[f"{die}_x"] = float(x)
                row[f"{die}_y"] = float(y)

            records_out.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(records_out, args.output, die_keys)

    unique_labels, counts = np.unique([r["predicted_label"] for r in records_out], return_counts=True)
    distribution = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    print(f"Reference layouts combined: {len(ref_y)}")
    print(f"Generated samples: {args.num_samples}")
    print(f"Output CSV: {args.output}")
    print(f"Predicted label distribution: {distribution}")


if __name__ == "__main__":
    main()
