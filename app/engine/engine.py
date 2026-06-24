"""Engine adapter — PythonValidationEngine is the canonical engine.

Rust citta_core is an optional extension for high-throughput English
text-pattern workloads only.  It covers 6 of 7 condition types but does
NOT handle numeric_comparison (requires CJK tokenisation and Chinese
number parsing) or knowledge-layer fields (source tracking).

The default engine is always PythonValidationEngine.  Rust is gated
behind CITTA_RUST_ENABLED=true and must be explicitly opted into.

For contract compliance and any numeric-threshold checks, use the
Python engine.  The Rust engine will return NOT_APPLICABLE for all
numeric_comparison rules.
"""
import logging
from app.config import settings

logger = logging.getLogger(__name__)

_engine_instance = None


class RustEngineAdapter:

    def __init__(self):
        import citta_core
        self._mod = citta_core
        required = ["load_package", "validate", "health"]
        for attr in required:
            if not hasattr(self._mod, attr):
                raise ImportError(f"citta_core missing: {attr}")
        logger.info("Rust engine adapter initialized (citta_core v%s)",
                    getattr(self._mod, "__version__", "unknown"))

    def load_package(self, package_data: dict) -> None:
        self._mod.load_package(package_data)

    def reload_package(self, package_id: str, new_package_data: dict) -> None:
        if hasattr(self._mod, "reload_package"):
            self._mod.reload_package(package_id, new_package_data)
        else:
            self.unload_package(package_id)
            self._mod.load_package(new_package_data)

    def unload_package(self, package_id: str) -> None:
        if hasattr(self._mod, "unload_package"):
            self._mod.unload_package(package_id)

    def list_packages(self) -> list:
        if hasattr(self._mod, "list_packages"):
            return self._mod.list_packages()
        return []

    def validate(self, input_data: dict, packages: list, options: dict) -> dict:
        return self._mod.validate(input_data, packages, options)

    def health(self) -> dict:
        return self._mod.health()


def get_engine():
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance
    if settings.RUST_ENABLED:
        try:
            _engine_instance = RustEngineAdapter()
            logger.info("Using Rust validation engine")
            return _engine_instance
        except (ImportError, Exception) as e:
            logger.warning("Rust engine unavailable (%s). Falling back to Python.", e)
    from app.engine.core import PythonValidationEngine
    _engine_instance = PythonValidationEngine(
        llm_url=settings.LLM_API_URL,
        llm_key=settings.LLM_API_KEY,
        llm_model=settings.LLM_MODEL,
    )
    logger.info("Using Python validation engine (full capability)")
    return _engine_instance


def reset_engine():
    global _engine_instance
    _engine_instance = None
