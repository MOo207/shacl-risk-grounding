"""
SHACL Validator Compatibility Shim

This module provides backward compatibility for imports.
Re-exports SHACLValidator from shacl_validator_neo4j.
"""

from .shacl_validator_neo4j import SHACLValidator

__all__ = ['SHACLValidator']
