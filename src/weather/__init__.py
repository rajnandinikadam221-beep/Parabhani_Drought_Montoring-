"""
Meteorological & SPI Processing Package
"""
from .openmeteo import fetch_weather_and_compute_spi, classify_spi

__all__ = ["fetch_weather_and_compute_spi", "classify_spi"]
