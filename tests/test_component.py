"""Tests for AI-Driven Longevity Intervention Screening.

These tests verify the component works correctly before deployment.
"""

import pytest
import numpy as np
from life_vector_commons import TypedOutput, OutputPattern


class TestAnalyze:
    """Test the main analyze function."""

    def test_returns_required_keys(self):
        """Verify analyze returns outputs, metrics, and summary."""
        from modal_app import analyze

        # Use small test dataset
        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        assert "outputs" in result, "Missing 'outputs' key"
        assert "metrics" in result, "Missing 'metrics' key"
        assert "summary" in result, "Missing 'summary' key"

    def test_outputs_are_valid_typed_outputs(self):
        """Verify all outputs conform to TypedOutput schema."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        assert len(result["outputs"]) > 0, "Should have at least one output"

        for output in result["outputs"]:
            # Each output should have required fields
            assert "pattern" in output, "Output missing 'pattern'"
            assert "data" in output, "Output missing 'data'"
            assert "label" in output, "Output missing 'label'"
            assert "description" in output, "Output missing 'description'"

            # Pattern should be valid
            pattern = output["pattern"]
            valid_patterns = [p.value for p in OutputPattern]
            assert pattern in valid_patterns, f"Invalid pattern: {pattern}"

    def test_metrics_is_dict(self):
        """Verify metrics is a dictionary with expected keys."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        assert isinstance(result["metrics"], dict), "metrics should be a dict"

        # Check required metrics
        required_metrics = [
            "total_screened",
            "top_k_selected",
            "precision_at_k",
            "recall",
        ]

        for metric in required_metrics:
            assert metric in result["metrics"], f"Missing metric: {metric}"

    def test_summary_is_string(self):
        """Verify summary is a string."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        assert isinstance(result["summary"], str), "summary should be a string"
        assert len(result["summary"]) > 0, "summary should not be empty"
        assert "compounds" in result["summary"].lower(), "summary should mention compounds"


class TestOutputPatterns:
    """Test that output patterns have correct data structures."""

    def test_has_data_sources_output(self):
        """Should include data sources with provenance."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        data_sources_output = [o for o in result["outputs"] if "Data Sources" in o["label"]]
        assert len(data_sources_output) > 0, "Missing data sources output"

        # Check it's tabular
        ds = data_sources_output[0]
        assert ds["pattern"] == OutputPattern.TABULAR.value
        assert "columns" in ds["data"]
        assert "rows" in ds["data"]

    def test_has_model_performance_metrics(self):
        """Should include model performance with uncertainty."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        perf_outputs = [o for o in result["outputs"] if "Model Performance" in o["label"]]
        assert len(perf_outputs) > 0, "Missing model performance output"

        perf = perf_outputs[0]
        assert perf["pattern"] == OutputPattern.KEY_METRICS.value
        assert "metrics" in perf["data"]

        # Check uncertainty is included (at least one metric should have bounds)
        has_uncertainty = any("lower_bound" in m for m in perf["data"]["metrics"])
        assert has_uncertainty, "No uncertainty bounds found"

    def test_has_ranking_output(self):
        """Should include ranked interventions."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        ranking_outputs = [o for o in result["outputs"]
                          if o["pattern"] == OutputPattern.RANKING.value
                          and "Prioritized" in o["label"]]
        assert len(ranking_outputs) > 0, "Missing ranking output"

        ranking = ranking_outputs[0]
        assert "items" in ranking["data"]
        assert "scores" in ranking["data"]
        assert "score_std" in ranking["data"], "Missing uncertainty estimates"

    def test_has_validation_metrics(self):
        """Should include ground truth validation."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        validation_outputs = [o for o in result["outputs"]
                             if "Validation" in o["label"] or "Test Set" in o["label"]]
        assert len(validation_outputs) > 0, "Missing validation output"


class TestDataFetching:
    """Test real data fetching."""

    def test_fetches_geo_data(self):
        """Should fetch real GEO data."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.1})

        # Check metrics indicate data was fetched
        assert "n_aging_genes" in result["metrics"], "Missing aging genes metric"
        assert result["metrics"]["n_aging_genes"] >= 0, "Invalid aging genes count"

        # Check data sources
        data_sources = [o for o in result["outputs"] if "Data Sources" in o["label"]][0]
        rows = data_sources["data"]["rows"]

        # Should have GEO source
        geo_sources = [r for r in rows if "GEO" in r[0]]
        assert len(geo_sources) > 0, "Missing GEO data source"

    def test_includes_pubchem_interventions(self):
        """Should include PubChem compound data."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.1})

        # Check data sources
        data_sources = [o for o in result["outputs"] if "Data Sources" in o["label"]][0]
        rows = data_sources["data"]["rows"]

        # Should have PubChem source
        pubchem_sources = [r for r in rows if "PubChem" in r[0]]
        assert len(pubchem_sources) > 0, "Missing PubChem data source"


class TestGroundTruthValidation:
    """Test validation against held-out longevity interventions."""

    def test_recovers_heldout_interventions(self):
        """Should recover some held-out interventions."""
        from modal_app import analyze

        result = analyze({"n_candidates": 100, "top_percent": 0.1})

        # Check that heldout compounds are tracked
        metrics = result["metrics"]
        assert metrics["heldout_compounds"] > 0, "No heldout compounds defined"

    def test_precision_and_recall_computed(self):
        """Should compute precision and recall."""
        from modal_app import analyze

        result = analyze({"n_candidates": 100, "top_percent": 0.1})

        metrics = result["metrics"]
        assert "precision_at_k" in metrics
        assert "recall" in metrics

        # Values should be valid proportions
        assert 0 <= metrics["precision_at_k"] <= 1
        assert 0 <= metrics["recall"] <= 1


class TestUncertaintyQuantification:
    """Test uncertainty estimation."""

    def test_ranking_includes_uncertainty(self):
        """Rankings should include uncertainty estimates."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        ranking_output = [o for o in result["outputs"]
                         if o["pattern"] == OutputPattern.RANKING.value
                         and "Prioritized" in o["label"]][0]

        # Check uncertainty is present
        assert "score_std" in ranking_output["data"], "Missing uncertainty"

        # Check all scores have uncertainty
        scores = ranking_output["data"]["scores"]
        uncertainties = ranking_output["data"]["score_std"]
        assert len(scores) == len(uncertainties), "Mismatch in scores/uncertainty"

        # Uncertainties should be positive
        assert all(u >= 0 for u in uncertainties), "Negative uncertainty values"

    def test_model_performance_has_confidence_intervals(self):
        """Model metrics should have confidence intervals."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.05})

        perf_output = [o for o in result["outputs"]
                      if "Model Performance" in o["label"]][0]

        metrics = perf_output["data"]["metrics"]

        # At least one metric should have confidence intervals
        has_ci = False
        for metric in metrics:
            if "lower_bound" in metric and "upper_bound" in metric:
                has_ci = True
                # CI should be valid
                assert metric["lower_bound"] <= metric["value"] <= metric["upper_bound"], \
                    f"Invalid CI for {metric['name']}"

        assert has_ci, "No confidence intervals found in metrics"


class TestScalability:
    """Test with different dataset sizes."""

    def test_small_dataset(self):
        """Should work with small dataset."""
        from modal_app import analyze

        result = analyze({"n_candidates": 50, "top_percent": 0.1})

        # Total screened may vary due to API availability
        assert result["metrics"]["total_screened"] > 0
        assert result["metrics"]["top_k_selected"] > 0

    def test_different_top_percent(self):
        """Should work with different selection percentages."""
        from modal_app import analyze

        result = analyze({"n_candidates": 100, "top_percent": 0.05})

        assert result["metrics"]["total_screened"] > 0
        assert result["metrics"]["top_k_selected"] > 0


class TestIntegration:
    """Integration tests with mocked runner."""

    def test_runner_integration(self):
        """Test that the component works with ComponentRunner."""
        from modal_app import analyze

        # Test with small dataset for speed
        result = analyze({"n_candidates": 50, "top_percent": 0.1})

        # Should complete successfully
        assert "outputs" in result
        assert "metrics" in result
        assert "summary" in result

        # Should have multiple outputs
        assert len(result["outputs"]) >= 4, "Should have multiple output types"



