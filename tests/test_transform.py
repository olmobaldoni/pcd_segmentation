"""Tests for the transform helpers used by the transformation gizmo."""

from __future__ import annotations

import numpy as np

from pcd_segmentation.core.transform import (
    apply_transform,
    extract_translation,
    relative_transform,
    rotate_vectors,
    translation_matrix,
)


def test_apply_transform():
    """Translations, rotations, and scales map points as expected."""
    translation = np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert np.allclose(apply_transform(points, translation), [[10.0, -5.0, 2.5], [11.0, -3.0, 5.5]])
    rotation = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(apply_transform(np.array([[1.0, 0.0, 0.0]]), rotation), [[0.0, 1.0, 0.0]])
    assert np.allclose(
        apply_transform(np.array([[1.0, 2.0, 3.0]]), np.diag([2.0, 2.0, 2.0, 1.0])),
        [[2.0, 4.0, 6.0]],
    )


def test_rotate_vectors():
    """Direction vectors ignore translation and use the inverse-transpose."""
    translated = np.array(
        [
            [1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(rotate_vectors(np.array([[1.0, 0.0, 0.0]]), translated), [[1.0, 0.0, 0.0]])
    nonuniform = np.diag([2.0, 1.0, 1.0, 1.0])
    assert np.allclose(rotate_vectors(np.array([[1.0, 0.0, 0.0]]), nonuniform), [[0.5, 0.0, 0.0]])


def test_translation_matrix():
    """The translation matrix maps the origin onto the centroid."""
    matrix = translation_matrix(np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
    assert np.allclose(extract_translation(matrix), [15.0, 0.0, 0.0])
    assert np.allclose(translation_matrix(np.empty((0, 3))), np.eye(4))


def test_relative_transform():
    """``relative_transform`` inverts the base like ``inv(base) @ combined``."""
    base = translation_matrix(np.array([[5.0, 5.0, 5.0]]))
    object_matrix = np.eye(4)
    object_matrix[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(relative_transform(base @ object_matrix, base), object_matrix)
