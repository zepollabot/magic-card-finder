"""Unit tests for AnalysisService parsing and helpers."""
import pytest

from app.services.analysis_service import AnalysisService


def test_parse_name_lines_one_per_line():
    """Each non-empty line is one card; no comma = no set hint."""
    out = AnalysisService._parse_name_lines(["Llanowar Elves", "Lightning Bolt", ""])
    assert out == [
        ("Llanowar Elves", None),
        ("Lightning Bolt", None),
    ]


def test_parse_name_lines_name_comma_set():
    """Line 'Name, Set' yields (name, set)."""
    out = AnalysisService._parse_name_lines(["Lightning Bolt, Dominaria", "Tarmogoyf, Modern Masters"])
    assert out == [
        ("Lightning Bolt", "Dominaria"),
        ("Tarmogoyf", "Modern Masters"),
    ]


def test_parse_name_lines_mixed():
    """Mix of plain names and 'Name, Set'."""
    out = AnalysisService._parse_name_lines([
        "Llanowar Elves",
        "Lightning Bolt, Foundations",
        "Birds of Paradise",
    ])
    assert out == [
        ("Llanowar Elves", None),
        ("Lightning Bolt", "Foundations"),
        ("Birds of Paradise", None),
    ]


def test_parse_name_lines_skips_empty():
    """Empty and whitespace-only lines are skipped."""
    out = AnalysisService._parse_name_lines(["  A  ", "", "  ", "B, SET"])
    assert out == [
        ("A", None),
        ("B", "SET"),
    ]
