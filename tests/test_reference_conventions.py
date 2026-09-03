"""Tests for the axis-convention definitions and their conversion math."""

from __future__ import annotations

import numpy as np
import pytest

from pcd_segmentation.reference import conventions as conv

SAMPLE = np.array(
    [
        [1.0, 2.0, 3.0],
        [-4.0, 5.0, -6.0],
        [0.5, -0.5, 0.25],
    ]
)


def test_preset_order_and_content():
    """NONE is first and the known triples are present."""
    presets = conv.preset_triples()
    assert presets[0] == conv.NONE_TRIPLE
    for triple in ("RUB", "RDF", "LUF", "FLU", "FRD"):
        assert triple in presets


def test_none_and_rub_are_identity():
    """NONE and RUB both describe the canonical right-handed frame."""
    assert np.allclose(conv.axis_matrix(conv.NONE_TRIPLE), np.eye(3))
    assert np.allclose(conv.axis_matrix("RUB"), np.eye(3))


def test_rdf_matrix():
    """RDF keeps X, negates Y and Z."""
    expected = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(conv.axis_matrix("RDF"), expected)


def test_known_conversion_rdf_to_rub():
    """Converting RDF to RUB maps (x, y, z) to (x, -y, -z)."""
    matrix = conv.axes_matrix("RDF", "RUB")
    expected = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(matrix, expected)
    result = conv.convert_points(SAMPLE, "RDF", "RUB")
    assert np.allclose(result, SAMPLE * [1.0, -1.0, -1.0])


def test_known_conversion_rdf_to_luf():
    """Converting RDF to LUF maps (x, y, z) to (-x, -y, z)."""
    matrix = conv.axes_matrix("RDF", "LUF")
    expected = np.diag([-1.0, -1.0, 1.0])
    assert np.allclose(matrix, expected)
    result = conv.convert_points(SAMPLE, "RDF", "LUF")
    assert np.allclose(result, SAMPLE * [-1.0, -1.0, 1.0])


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("RUB", "RDF"),
        ("RDF", "LUF"),
        ("FLU", "FRD"),
        ("FRD", "RUB"),
        ("LUF", "FLU"),
    ],
)
def test_round_trip_is_identity(source, target):
    """Converting there and back reproduces the original coordinates."""
    forward = conv.axes_matrix(source, target)
    backward = conv.axes_matrix(target, source)
    assert np.allclose(forward @ backward, np.eye(3), atol=1e-12)
    round_trip = conv.convert_points(conv.convert_points(SAMPLE, source, target), target, source)
    assert np.allclose(round_trip, SAMPLE, atol=1e-12)


def test_handedness():
    """With the canonical mapping all offered triples are right-handed."""
    for triple in ("RUB", "RDF", "LUF", "FLU", "FRD"):
        assert conv.handedness(triple) == "right"
    assert conv.handedness(conv.NONE_TRIPLE) == "right"


def test_conversion_is_norm_preserving():
    """Axis conversions are signed permutations, so they keep distances."""
    result = conv.convert_points(SAMPLE, "FLU", "RDF")
    assert np.allclose(np.linalg.norm(result, axis=1), np.linalg.norm(SAMPLE, axis=1))


@pytest.mark.parametrize(
    ("triple", "expected"),
    [
        ("RUB", "x=right, y=up, z=back"),
        ("RDF", "x=right, y=down, z=forward"),
        ("LUF", "x=left, y=up, z=forward"),
        ("FLU", "x=forward, y=left, z=up"),
        ("FRD", "x=forward, y=right, z=down"),
    ],
)
def test_axis_letters(triple, expected):
    """The axis letters name the world direction of each data axis."""
    assert conv.axis_letters(triple) == expected


def test_axis_letters_none_is_empty():
    """NONE has no axis letters."""
    assert conv.axis_letters(conv.NONE_TRIPLE) == ""


@pytest.mark.parametrize(
    "function", [conv.axis_matrix, conv.description, conv.handedness, conv.axis_letters]
)
def test_unknown_triple_raises(function):
    """Unknown triple keys raise a ValueError."""
    with pytest.raises(ValueError, match="Unknown"):
        function("XYZ")


def test_convert_preserves_shape():
    """convert_points returns a copy with the same shape."""
    result = conv.convert_points(SAMPLE, "NONE", "NONE")
    assert result.shape == SAMPLE.shape
    assert np.allclose(result, SAMPLE)
    assert result is not SAMPLE
