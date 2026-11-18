"""Discriminant analysis based on Kohonen statistics (Komarov & Kuravsky, MDA 2023)."""

from .classifier import KohonenDA, sturges_bins

__all__ = ["KohonenDA", "sturges_bins"]
