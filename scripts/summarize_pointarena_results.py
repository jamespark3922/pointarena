#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path


DISPLAY_NAMES = {
    "affordable": "Affordance",
    "spatial": "Spatial",
    "reasoning": "Reasoning",
    "steerable": "Steerability",
    "counting": "Counting",
}


def load_data(data_file):
    with open(data_file) as f:
        return json.load(f)


def summarize(results, data):
    category_by_image = {item["image_filename"]: item.get("category", "") for item in data}
    summary = {
        display: {"success": 0, "total": 0, "accuracy": 0.0}
        for display in DISPLAY_NAMES.values()
    }
    details = results.get("details", [])
    for idx, detail in enumerate(details):
        category = detail.get("category") or category_by_image.get(detail.get("image", ""))
        if not category and idx < len(data):
            category = data[idx].get("category", "")
        display = DISPLAY_NAMES.get(category)
        if not display:
            continue
        summary[display]["total"] += 1
        if detail.get("success"):
            summary[display]["success"] += 1

    for stats in summary.values():
        if stats["total"]:
            stats["accuracy"] = 100.0 * stats["success"] / stats["total"]

    total = sum(stats["total"] for stats in summary.values())
    success = sum(stats["success"] for stats in summary.values())
    summary["Average"] = {
        "success": success,
        "total": total,
        "accuracy": 100.0 * success / total if total else 0.0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize PointArena results by leaderboard category")
    parser.add_argument("results_file", help="Path to static_results/*.json")
    parser.add_argument("--data", default="data.json", help="PointArena data.json path")
    args = parser.parse_args()

    results_file = Path(args.results_file)
    with open(results_file) as f:
        results = json.load(f)
    data = load_data(args.data)
    summary = summarize(results, data)

    total = results.get("total", len(results.get("details", [])))
    success = results.get("success", sum(1 for d in results.get("details", []) if d.get("success")))
    if total != len(data):
        print(f"Warning: result file is partial: {total}/{len(data)} examples.")
    print(f"Overall from result counters: {success}/{total} = {100.0 * success / total if total else 0.0:.2f}%")

    out_json = results_file.with_name(results_file.stem + "_category_summary.json")
    out_csv = results_file.with_name(results_file.stem + "_category_summary.csv")
    print(" | ".join(f"{cat}: {stats['accuracy']:.2f}%" for cat, stats in summary.items()))
    try:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Success", "Total", "Accuracy"])
            for category, stats in summary.items():
                writer.writerow([category, stats["success"], stats["total"], f"{stats['accuracy']:.2f}"])
        print(f"Wrote {out_json}")
        print(f"Wrote {out_csv}")
    except PermissionError as e:
        print(f"Could not write summary files: {e}")


if __name__ == "__main__":
    main()
