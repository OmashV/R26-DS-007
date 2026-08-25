import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.detector_service import DetectorService


def main() -> None:
    detectors = DetectorService.from_project_defaults(
        PROJECT_ROOT
    )

    cases = [
        {
            "name": "reasoning-like",
            "previous": "I think it is 10.",
            "current": (
                "Because 20 percent of 50 is 10."
            ),
        },
        {
            "name": "uncertain",
            "previous": None,
            "current": (
                "I'm not sure, maybe the answer is 12."
            ),
        },
        {
            "name": "clarification",
            "previous": None,
            "current": (
                "Can you explain what you mean by denominator?"
            ),
        },
        {
            "name": "short answer",
            "previous": None,
            "current": "10",
        },
    ]

    print("=" * 72)
    print("REAL DETECTOR SMOKE TEST")
    print("=" * 72)

    for case in cases:
        result = detectors.predict_all(
            current_text=case["current"],
            previous_student_text=case["previous"],
        )

        print(f"\n[{case['name']}]")
        print(f"previous: {case['previous']}")
        print(f"current : {case['current']}")
        print(
            "reasoning     : "
            f"{result['reasoning_probability']:.4f} "
            f"(present={result['reasoning_present']})"
        )
        print(
            "uncertainty   : "
            f"{result['uncertainty_probability']:.4f} "
            f"(present={result['uncertainty_present']})"
        )
        print(
            "clarification : "
            f"{result['clarification_probability']:.4f} "
            f"(present={result['clarification_present']})"
        )

    print("\nLoaded artifacts successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
