"""Citta Engine exception types.
Moved from app.main to break circular imports between main.py and routes.
"""
class CittaError(Exception):
    """Base error with HTTP status code and error code."""
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
class PackageNotFoundError(CittaError):
    def __init__(self, package_id: str):
        super().__init__("PACKAGE_NOT_FOUND", f"Package '{package_id}' is not loaded.", 404)
class PackageAlreadyLoadedError(CittaError):
    def __init__(self, package_id: str):
        super().__init__(
            "PACKAGE_ALREADY_LOADED",
            f"Package '{package_id}' is already loaded. Use PUT to update.",
            409,
        )
class InvalidRuleSchemaError(CittaError):
    def __init__(self, message: str):
        super().__init__("INVALID_RULE_SCHEMA", message, 400)
class CompilationError(CittaError):
    def __init__(self, message: str):
        super().__init__("COMPILATION_ERROR", message, 422)
class InputTooLargeError(CittaError):
    def __init__(self, chars: int, limit: int):
        super().__init__(
            "INPUT_TOO_LARGE",
            f"Input exceeds maximum of {limit} characters (got {chars}).",
            400,
        )
class TimeoutError(CittaError):
    def __init__(self):
        super().__init__("TIMEOUT", "Validation exceeded timeout_ms.", 408)
class RateLimitedError(CittaError):
    def __init__(self):
        super().__init__("RATE_LIMITED", "Too many requests. Please slow down.", 429)
class InternalError(CittaError):
    def __init__(self, message: str = "Unexpected engine error."):
        super().__init__("INTERNAL", message, 500)
