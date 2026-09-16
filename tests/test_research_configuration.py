"""Research policy boundaries shared by every command and experiment."""

import pytest

from regime_alloc.config import load_config
from regime_alloc.contracts import ResearchError


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_forward_fill_stays_within_two_calendar_months(tmp_path, limit):
    path = tmp_path / "research.toml"
    path.write_text(f"[research]\nffill_limit = {limit}\n", encoding="utf-8")
    assert load_config(path).research.ffill_limit == limit


def test_longer_forward_fill_is_not_a_supported_research_policy(tmp_path):
    path = tmp_path / "research.toml"
    path.write_text("[research]\nffill_limit = 3\n", encoding="utf-8")
    with pytest.raises(ResearchError, match="invalid_config"):
        load_config(path)


@pytest.mark.parametrize("threshold", [.90, .95, .99])
def test_pca_sensitivity_thresholds_remain_configurable(tmp_path, threshold):
    path = tmp_path / "research.toml"
    path.write_text(f"[research]\npca_variance = {threshold}\n", encoding="utf-8")
    assert load_config(path).research.pca_variance == threshold


@pytest.mark.parametrize("count", [4, 6])
def test_research_uses_five_normal_regimes(tmp_path, count):
    path = tmp_path / "research.toml"
    path.write_text(f"[research]\nk_normal = {count}\n", encoding="utf-8")
    with pytest.raises(ResearchError, match="invalid_config"):
        load_config(path)
