"""Persistência SQLite e autorização dos fluxos administrativos sintéticos."""

from src.database.sqlite import (
    AuthorizationError,
    ClinicDatabase,
    ConflictError,
    Principal,
    Role,
)

__all__ = ["AuthorizationError", "ClinicDatabase", "ConflictError", "Principal", "Role"]
