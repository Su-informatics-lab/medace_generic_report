#!/usr/bin/env python3
"""Batch extraction driver for pediatric genetic reports using the Hugging Face backend."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from main import HFBackend, extract, load_report, resolve_model
from schema import GeneticReportExtraction, get_extraction_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract pediatric genetic reports with the HF backend"
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-dir", type=Path, required=True)
    parser.add_argument("--status-tsv", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, default=None)
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="oss-120b")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "text", "pdf"],
        default="text",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=12000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                out.update(flatten(value, new_key))
            elif isinstance(value, list):
                out[new_key] = json.dumps(value, ensure_ascii=False)
            else:
                out[new_key] = value
    else:
        out[prefix or "value"] = obj
    return out


def discover_files(input_path: Path, pattern: str, limit: int) -> list[Path]:
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(path for path in input_path.glob(pattern) if path.is_file())
    if limit > 0:
        files = files[:limit]
    return files


def infer_input_format(requested: str) -> str | None:
    if requested == "auto":
        return None
    return requested


def ensure_report_file_name(
    result: GeneticReportExtraction,
    source_path: Path,
) -> GeneticReportExtraction:
    if result.test_info.file:
        return result
    test_info = result.test_info.model_copy(update={"file": source_path.name})
    return result.model_copy(update={"test_info": test_info})


def write_aggregate_csv(json_paths: list[Path], csv_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames = ["source_file", "json_path"]

    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        row = {
            "source_file": payload.get("test_info", {}).get("file")
            or json_path.stem.replace(".extracted", ""),
            "json_path": str(json_path),
        }
        row.update(flatten(payload))
        rows.append(row)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.fail_dir.mkdir(parents=True, exist_ok=True)
    args.status_tsv.parent.mkdir(parents=True, exist_ok=True)
    if args.aggregate_csv is not None:
        args.aggregate_csv.parent.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.input_path, args.pattern, args.limit)
    if not files:
        raise SystemExit(
            f"no files found under {args.input_path} with pattern {args.pattern}"
        )

    model_id = resolve_model(args.model)
    print(f"[info] loading HF model once: {args.model} -> {model_id}", flush=True)
    backend = HFBackend(model_id, num_gpus=args.num_gpus, dtype=args.dtype)
    model_cls = get_extraction_model()

    ok = 0
    failed = 0
    written_jsons: list[Path] = []

    with args.status_tsv.open("w", encoding="utf-8", newline="") as status_handle:
        status_handle.write("source_file\tjson_path\tstatus\tmessage\n")

        for index, path in enumerate(files, start=1):
            out_path = args.output_dir / f"{path.stem}.extracted.json"
            err_path = args.fail_dir / f"{path.stem}.error.txt"

            if args.skip_existing and out_path.exists():
                print(f"[skip] {index}/{len(files)} {path.name}", flush=True)
                status_handle.write(f"{path}\t{out_path}\tskip\t\n")
                written_jsons.append(out_path)
                continue

            print(f"[run] {index}/{len(files)} {path.name}", flush=True)
            try:
                report_text = load_report(path, infer_input_format(args.input_format))
                result = extract(
                    report_text,
                    backend,
                    model_cls=model_cls,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_retries=args.max_retries,
                )
                if not isinstance(result, GeneticReportExtraction):
                    result = GeneticReportExtraction.model_validate(
                        result.model_dump(mode="json")
                    )
                result = ensure_report_file_name(result, path)
                out_path.write_text(
                    json.dumps(
                        result.model_dump(mode="json"),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                if err_path.exists():
                    err_path.unlink()
                status_handle.write(f"{path}\t{out_path}\tok\t\n")
                written_jsons.append(out_path)
                ok += 1
                print(f"[ok] {path.name} -> {out_path.name}", flush=True)
            except Exception as exc:
                failed += 1
                err_path.write_text(
                    f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                status_handle.write(f"{path}\t{out_path}\tfail\t{err_path}\n")
                print(f"[fail] {path.name}: {exc}", file=sys.stderr, flush=True)

    if args.aggregate_csv is not None:
        write_aggregate_csv(written_jsons, args.aggregate_csv)
        print(f"[csv] wrote {args.aggregate_csv}", flush=True)

    print(
        f"[done] ok={ok} failed={failed} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
