"""
Data validation suite — equivalent to Great Expectations, implemented
as standalone validators with clear pass/fail reporting.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import FEATURE_RANGES, REQUIRED_INFERENCE_FEATURES, CROP_LABELS


class ValidationResult:
    """Container for a single validation check result."""

    def __init__(self, name: str, passed: bool, severity: str = "critical",
                 details: str = "", stats: dict = None):
        self.name = name
        self.passed = passed
        self.severity = severity  # "critical" | "warning" | "info"
        self.details = details
        self.stats = stats or {}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "details": self.details,
            "stats": self.stats,
        }


class DataValidationSuite:
    """
    A comprehensive data validation suite for crop recommendation data.
    Checks ranges, nulls, cardinalities, and schema compliance.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: List[ValidationResult] = []

    def check_required_columns(self) -> "DataValidationSuite":
        """Verify all required inference columns exist."""
        missing = [c for c in REQUIRED_INFERENCE_FEATURES if c not in self.df.columns]
        self.results.append(ValidationResult(
            name="required_columns_present",
            passed=len(missing) == 0,
            severity="critical",
            details=f"Missing columns: {missing}" if missing else "All required columns present",
            stats={"missing_columns": missing, "total_required": len(REQUIRED_INFERENCE_FEATURES)},
        ))
        return self

    def check_no_all_null_columns(self) -> "DataValidationSuite":
        """No column should be entirely null."""
        all_null = [c for c in self.df.columns if self.df[c].isna().all()]
        self.results.append(ValidationResult(
            name="no_all_null_columns",
            passed=len(all_null) == 0,
            severity="warning",
            details=f"All-null columns: {all_null}" if all_null else "No all-null columns",
            stats={"all_null_columns": all_null},
        ))
        return self

    def check_null_rates(self, max_null_pct: float = 0.5) -> "DataValidationSuite":
        """Required features should have null rate < max_null_pct."""
        high_null = {}
        for col in REQUIRED_INFERENCE_FEATURES:
            if col in self.df.columns:
                null_pct = self.df[col].isna().mean()
                if null_pct > max_null_pct:
                    high_null[col] = float(null_pct)
        self.results.append(ValidationResult(
            name="null_rates_acceptable",
            passed=len(high_null) == 0,
            severity="critical",
            details=f"High null columns: {high_null}" if high_null else "All null rates acceptable",
            stats={"high_null_columns": high_null, "threshold": max_null_pct},
        ))
        return self

    def check_feature_ranges(self) -> "DataValidationSuite":
        """Numeric features should be within expected ranges."""
        violations = {}
        for col, (lo, hi) in FEATURE_RANGES.items():
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64, np.float32]:
                vals = self.df[col].dropna()
                below = (vals < lo).sum()
                above = (vals > hi).sum()
                if below > 0 or above > 0:
                    violations[col] = {
                        "below_min": int(below),
                        "above_max": int(above),
                        "expected_range": [lo, hi],
                        "actual_range": [float(vals.min()), float(vals.max())],
                    }
        self.results.append(ValidationResult(
            name="feature_ranges_valid",
            passed=len(violations) == 0,
            severity="critical",
            details=f"Range violations in {list(violations.keys())}" if violations else "All features in range",
            stats={"violations": violations},
        ))
        return self

    def check_target_cardinality(self) -> "DataValidationSuite":
        """Target crop should have known labels."""
        if "target_crop" not in self.df.columns:
            self.results.append(ValidationResult(
                name="target_cardinality",
                passed=False, severity="critical",
                details="target_crop column missing",
            ))
            return self

        unknown = set(self.df["target_crop"].dropna().unique()) - set(CROP_LABELS)
        self.results.append(ValidationResult(
            name="target_cardinality",
            passed=len(unknown) == 0,
            severity="warning",
            details=f"Unknown crops: {unknown}" if unknown else f"All {self.df['target_crop'].nunique()} crops recognized",
            stats={"unknown_crops": list(unknown), "n_unique": int(self.df["target_crop"].nunique())},
        ))
        return self

    def check_no_duplicate_samples(self) -> "DataValidationSuite":
        """Check for duplicate sample_ids."""
        if "sample_id" in self.df.columns:
            n_dup = self.df["sample_id"].duplicated().sum()
            self.results.append(ValidationResult(
                name="no_duplicate_sample_ids",
                passed=n_dup == 0,
                severity="warning",
                details=f"{n_dup} duplicate sample_ids" if n_dup else "No duplicates",
                stats={"n_duplicates": int(n_dup)},
            ))
        return self

    def check_class_balance(self, min_samples_per_class: int = 10) -> "DataValidationSuite":
        """Each crop class should have minimum samples."""
        if "target_crop" not in self.df.columns:
            return self
        counts = self.df["target_crop"].value_counts()
        under = counts[counts < min_samples_per_class]
        self.results.append(ValidationResult(
            name="class_balance_minimum",
            passed=len(under) == 0,
            severity="warning",
            details=f"Underrepresented classes: {dict(under)}" if len(under) else "All classes have sufficient samples",
            stats={
                "min_samples": int(counts.min()),
                "max_samples": int(counts.max()),
                "threshold": min_samples_per_class,
            },
        ))
        return self

    def check_location_ids(self) -> "DataValidationSuite":
        """Verify location_id column exists and is populated."""
        if "location_id" not in self.df.columns:
            self.results.append(ValidationResult(
                name="location_ids_present",
                passed=False, severity="critical",
                details="location_id column missing",
            ))
            return self
        n_locations = self.df["location_id"].nunique()
        null_pct = self.df["location_id"].isna().mean()
        self.results.append(ValidationResult(
            name="location_ids_present",
            passed=null_pct == 0 and n_locations >= 2,
            severity="critical",
            details=f"{n_locations} unique locations, {null_pct:.1%} null",
            stats={"n_locations": int(n_locations), "null_pct": float(null_pct)},
        ))
        return self

    def run_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        (
            self.check_required_columns()
            .check_no_all_null_columns()
            .check_null_rates()
            .check_feature_ranges()
            .check_target_cardinality()
            .check_no_duplicate_samples()
            .check_class_balance()
            .check_location_ids()
        )
        return self.results

    def report(self) -> dict:
        """Generate validation report."""
        if not self.results:
            self.run_all()

        critical_failures = sum(
            1 for r in self.results if not r.passed and r.severity == "critical"
        )
        warnings = sum(
            1 for r in self.results if not r.passed and r.severity == "warning"
        )
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        report = {
            "summary": {
                "total_checks": total,
                "passed": passed,
                "critical_failures": critical_failures,
                "warnings": warnings,
                "overall_status": "PASS" if critical_failures == 0 else "FAIL",
            },
            "checks": [r.to_dict() for r in self.results],
        }
        return report


def validate_data(data_path: str, output_path: str = None) -> dict:
    """Load data and run full validation suite."""
    print("=" * 60)
    print("DATA VALIDATION SUITE")
    print("=" * 60)

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {data_path}")

    suite = DataValidationSuite(df)
    report = suite.report()

    # Print results
    for check in report["checks"]:
        status = "✓" if check["passed"] else "✗"
        sev = check["severity"].upper()
        print(f"  [{status}] [{sev:8s}] {check['name']}: {check['details']}")

    summary = report["summary"]
    print(f"\nOverall: {summary['passed']}/{summary['total_checks']} passed, "
          f"{summary['critical_failures']} critical failures, "
          f"{summary['warnings']} warnings")
    print(f"Status: {summary['overall_status']}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")

    print("=" * 60)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data validation")
    parser.add_argument("--data", default="data/processed/crop_recommendation_clean.parquet")
    parser.add_argument("--output", default="data/processed/validation_report.json")
    args = parser.parse_args()
    validate_data(args.data, args.output)
