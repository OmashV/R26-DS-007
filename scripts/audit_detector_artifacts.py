from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXTENSIONS = {".joblib", ".pkl", ".pickle"}

SEARCH_TERMS = [
    "TfidfVectorizer",
    "LogisticRegression",
    "joblib.dump",
    "pickle.dump",
    "reasoning",
    "uncertainty",
    "clarification",
    "[SEP]",
    "text_a",
    "text_b",
    "ColumnTransformer",
    "Pipeline",
]


def describe_model(obj: Any) -> dict:
    info = {
        "type": f"{type(obj).__module__}.{type(obj).__name__}",
    }

    if hasattr(obj, "classes_"):
        try:
            info["classes"] = [str(x) for x in obj.classes_]
        except Exception:
            pass

    if hasattr(obj, "steps"):
        try:
            info["pipeline_steps"] = [
                {
                    "name": name,
                    "type": f"{type(step).__module__}.{type(step).__name__}",
                }
                for name, step in obj.steps
            ]
        except Exception:
            pass

    if hasattr(obj, "named_steps"):
        try:
            info["named_steps"] = list(obj.named_steps.keys())
        except Exception:
            pass

    if hasattr(obj, "transformers"):
        try:
            info["column_transformers"] = [
                {
                    "name": name,
                    "transformer_type": (
                        f"{type(transformer).__module__}."
                        f"{type(transformer).__name__}"
                    ),
                    "columns": str(columns),
                }
                for name, transformer, columns in obj.transformers
            ]
        except Exception:
            pass

    return info


def find_model_artifacts() -> list[dict]:
    results = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in MODEL_EXTENSIONS:
            continue

        record = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": path.stat().st_size,
        }

        try:
            model = joblib.load(path)
            record["load_ok"] = True
            record["model"] = describe_model(model)
        except Exception as exc:
            record["load_ok"] = False
            record["load_error"] = f"{type(exc).__name__}: {exc}"

        results.append(record)

    return sorted(results, key=lambda x: x["path"].lower())


def relevant_source_files() -> list[Path]:
    candidates = []

    for folder_name in (
        "src",
        "scripts",
        "notebooks",
        "experiments",
        "training",
    ):
        folder = PROJECT_ROOT / folder_name
        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
                ".py",
                ".ipynb",
                ".md",
                ".txt",
            }:
                candidates.append(path)

    return candidates


def search_training_references() -> list[dict]:
    matches = []

    for path in relevant_source_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lines = text.splitlines()
        line_matches = []

        for i, line in enumerate(lines, start=1):
            if any(term.lower() in line.lower() for term in SEARCH_TERMS):
                cleaned = re.sub(r"\s+", " ", line).strip()
                if len(cleaned) > 240:
                    cleaned = cleaned[:237] + "..."

                line_matches.append(
                    {
                        "line": i,
                        "text": cleaned,
                    }
                )

        if line_matches:
            matches.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "matches": line_matches[:80],
                }
            )

    return matches


def classify_candidates(model_artifacts: list[dict]) -> dict:
    buckets = {
        "reasoning": [],
        "uncertainty": [],
        "clarification": [],
        "unclassified": [],
    }

    for artifact in model_artifacts:
        lower = artifact["path"].lower()
        matched = False

        for key in ("reasoning", "uncertainty", "clarification"):
            if key in lower:
                buckets[key].append(artifact["path"])
                matched = True

        if not matched:
            buckets["unclassified"].append(artifact["path"])

    return buckets


def main() -> None:
    models = find_model_artifacts()
    source_refs = search_training_references()

    report = {
        "project_root": str(PROJECT_ROOT),
        "model_artifacts": models,
        "candidate_models": classify_candidates(models),
        "training_references": source_refs,
    }

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "detector_artifact_audit.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("DETECTOR ARTIFACT AUDIT")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model artifacts found: {len(models)}")

    print("\nCandidate model paths:")
    for key, paths in report["candidate_models"].items():
        print(f"\n  {key}:")
        if paths:
            for path in paths:
                print(f"    - {path}")
        else:
            print("    (none found by filename)")

    print("\nLoaded model details:")
    for model in models:
        status = "OK" if model.get("load_ok") else "FAILED"
        print(f"\n  [{status}] {model['path']}")

        if model.get("load_ok"):
            model_info = model.get("model", {})
            print(f"    type: {model_info.get('type')}")
            if "classes" in model_info:
                print(f"    classes: {model_info['classes']}")
            if "pipeline_steps" in model_info:
                print("    pipeline steps:")
                for step in model_info["pipeline_steps"]:
                    print(f"      - {step['name']}: {step['type']}")
        else:
            print(f"    error: {model.get('load_error')}")

    print(
        f"\nTraining/source files with relevant references: "
        f"{len(source_refs)}"
    )

    for item in source_refs[:30]:
        print(f"\n  {item['path']}")
        for match in item["matches"][:12]:
            print(f"    L{match['line']}: {match['text']}")

    print("\nFull JSON report:")
    print(f"  {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
