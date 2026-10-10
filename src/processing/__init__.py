"""
Geospatial & Hydro-Climatic Processing Package
"""
from .indices import process_monthly_indices
from .classify import compute_drought_stages_and_cdi

__all__ = ["process_monthly_indices", "compute_drought_stages_and_cdi"]
