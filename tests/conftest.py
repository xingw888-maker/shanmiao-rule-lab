"""Shared fixtures for handler unit tests.

Provides:
    make_matcher()  -- factory: PythonMatcher with empty _structured_inputs
    make_rule()     -- factory: CompiledRule with minimal boilerplate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_matcher():
    """Return a PythonMatcher instance with empty _structured_inputs."""
    from app.engine.core import PythonMatcher
    m = PythonMatcher(llm_url="", llm_key="", llm_model="")
    if not hasattr(m, "_structured_inputs"):
        m._structured_inputs = {}
    return m


def make_rule(
    rule_id="test-rule",
    name="test",
    condition_type="numeric_comparison",
    condition_params=None,
    severity="error",
    message="",
    category="",
    version="1",
    package_id="p1",
    package_version="1",
    **kwargs,
):
    """Factory: build a CompiledRule with minimal boilerplate."""
    from app.engine.core import CompiledRule
    cp = condition_params if condition_params is not None else {}
    return CompiledRule(
        id=rule_id,
        name=name,
        condition_type=condition_type,
        condition_params=cp,
        severity=severity,
        message=message,
        category=category,
        version=version,
        package_id=package_id,
        package_version=package_version,
        **kwargs,
    )


# pytest fixture compatibility (optional)
try:
    import pytest

    @pytest.fixture
    def matcher():
        return make_matcher()
except ImportError:
    pass
