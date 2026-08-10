"""
Data Ingestion Pipeline for INTERSYMBOLIC-GRC

This pipeline loads, parses, and ingests multi-source cybersecurity event data
into the Asset Relationship Graph (ARG) with SHACL validation and data quality checks.

Supported Data Sources:
- CSE-CIC-IDS2018: AWS network traffic with attack scenarios
- LANL Comprehensive: Authentication, network, DNS, process events
"""

__version__ = "0.1.0"
__author__ = "INTERSYMBOLIC-GRC Team"
