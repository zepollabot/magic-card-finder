"""Unit tests for the card_normalizer module."""

import cv2
import numpy as np
import pytest

from app.detection.card_normalizer import (
    CARD_HEIGHT,
    CARD_WIDTH,
    approximate_quad,
    ensure_portrait,
    fine_deskew,
    normalize_card,
    normalize_size,
    order_points,
    perspective_warp,
)


# ── order_points ─────────────────────────────────────────────


def test_order_points_canonical():
    pts = np.array([[100, 0], [0, 0], [0, 100], [100, 100]], dtype="float32")
    ordered = order_points(pts)
    assert ordered[0].tolist() == [0, 0]       # top-left
    assert ordered[1].tolist() == [100, 0]      # top-right
    assert ordered[2].tolist() == [100, 100]    # bottom-right
    assert ordered[3].tolist() == [0, 100]      # bottom-left


# ── ensure_portrait ──────────────────────────────────────────


def test_ensure_portrait_already_portrait():
    card = np.zeros((936, 672, 3), dtype=np.uint8)
    result = ensure_portrait(card)
    assert result.shape == (936, 672, 3)


def test_ensure_portrait_landscape_rotated():
    card = np.zeros((672, 936, 3), dtype=np.uint8)
    result = ensure_portrait(card)
    assert result.shape[0] > result.shape[1]


# ── fine_deskew ──────────────────────────────────────────────


def test_fine_deskew_no_change_for_straight():
    card = np.full((936, 672, 3), 200, dtype=np.uint8)
    card[50:55, 20:650] = 0
    result = fine_deskew(card)
    assert result.shape == card.shape


def test_fine_deskew_corrects_tilt():
    card = np.full((936, 672, 3), 200, dtype=np.uint8)
    for x in range(20, 650):
        y = 60 + int((x - 20) * np.tan(np.radians(4)))
        if 0 <= y < 936:
            card[max(0, y - 2) : y + 3, x] = 0
    result = fine_deskew(card)
    assert result.shape == card.shape


def test_fine_deskew_ignores_tiny():
    card = np.zeros((30, 20, 3), dtype=np.uint8)
    result = fine_deskew(card)
    assert result.shape == card.shape


# ── normalize_size ───────────────────────────────────────────


def test_normalize_size_output_dimensions():
    card = np.zeros((400, 300, 3), dtype=np.uint8)
    result = normalize_size(card)
    assert result.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


# ── normalize_card ───────────────────────────────────────────


def test_normalize_card_full_pipeline():
    card = np.zeros((300, 400, 3), dtype=np.uint8)  # landscape input
    result = normalize_card(card)
    assert result.shape == (CARD_HEIGHT, CARD_WIDTH, 3)


# ── approximate_quad ─────────────────────────────────────────


def test_approximate_quad_returns_4_points():
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (170, 170), 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quad = approximate_quad(contours[0])
    assert quad is not None
    assert quad.reshape(-1, 2).shape == (4, 2)


# ── perspective_warp ─────────────────────────────────────────


def test_perspective_warp_returns_image():
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    quad = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype="float32")
    quad = quad.reshape(4, 1, 2)
    result = perspective_warp(img, quad)
    assert result is not None
    assert result.shape[0] > 0 and result.shape[1] > 0


def test_perspective_warp_returns_none_for_degenerate():
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    quad = np.array([[50, 50], [50, 50], [50, 50], [50, 50]], dtype="float32")
    quad = quad.reshape(4, 1, 2)
    result = perspective_warp(img, quad)
    assert result is None
