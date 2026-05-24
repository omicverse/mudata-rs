"""pymudata — PyO3 bindings for mudata-rs.

Mirrors :mod:`pyanndata` for the multimodal AnnData (``MuData``) on-disk
format. Exposes the compiled Rust extension under
:mod:`pymudata._backend`; high-level Python users see :class:`MuData`
and :func:`read_h5mu` directly at the package root.
"""

from ._backend import MuData, read_h5mu  # type: ignore[attr-defined]

__version__ = "0.1.0"

__all__ = ["MuData", "read_h5mu", "__version__"]
