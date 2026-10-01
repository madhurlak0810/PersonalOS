"""PersonalOS Agent Framework."""

__version__ = "0.1.0"
__author__ = "PersonalOS Contributors"

# Installed at import so it cannot be forgotten by an entry point: every
# process that loads this package redacts credentials from its log records
# before any handler sees them. See `personalos.observability.log_redaction`.
from personalos.observability.log_redaction import install_log_redaction  # noqa: E402

install_log_redaction()
