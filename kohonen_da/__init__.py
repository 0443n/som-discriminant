"""Discriminant analysis based on Kohonen statistics (Komarov & Kuravsky, MDA 2023)."""

from .classifier import KohonenDA, sturges_bins
from .v2 import KohonenDA2
from .v3 import KohonenDA3, KohonenDA3Auto

__all__ = ["KohonenDA", "KohonenDA2", "KohonenDA3", "KohonenDA3Auto", "sturges_bins"]
