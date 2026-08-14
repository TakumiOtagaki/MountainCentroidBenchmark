"""Evaluation helpers for references using round and square brackets."""

from __future__ import annotations


_OPEN_TO_CLOSE = {"(": ")", "[": "]"}
_CLOSE_TO_OPEN = {closing: opening for opening, closing in _OPEN_TO_CLOSE.items()}


def pairs_from_extended_dot_bracket(structure: str) -> list[tuple[int, int]]:
    """Return 1-based pair endpoints, parsing ``()`` and ``[]`` independently."""
    stacks = {opening: [] for opening in _OPEN_TO_CLOSE}
    pairs: list[tuple[int, int]] = []
    for position, character in enumerate(structure, start=1):
        if character == ".":
            continue
        if character in _OPEN_TO_CLOSE:
            stacks[character].append(position)
            continue
        if character in _CLOSE_TO_OPEN:
            opening = _CLOSE_TO_OPEN[character]
            if not stacks[opening]:
                raise ValueError(
                    f"Unbalanced closing bracket at position {position}: {character}"
                )
            pairs.append((stacks[opening].pop(), position))
            continue
        raise ValueError(f"Unknown dot-bracket character: {character}")
    unclosed = [position for stack in stacks.values() for position in stack]
    if unclosed:
        raise ValueError(f"Unbalanced opening brackets at positions {sorted(unclosed)}")
    return sorted(pairs)


def mountain_heights(structure: str) -> tuple[int, ...]:
    """Return the number of base pairs spanning each sequence boundary."""
    differences = [0] * (len(structure) + 1)
    for left, right in pairs_from_extended_dot_bracket(structure):
        differences[left] += 1
        differences[right] -= 1
    height = 0
    heights: list[int] = []
    for boundary in range(1, len(structure)):
        height += differences[boundary]
        heights.append(height)
    return tuple(heights)


def base_pair_f1(predicted: str, reference: str) -> float:
    """Return base-pair F1 after matching pairs by their endpoints."""
    if len(predicted) != len(reference):
        raise ValueError("Structures must have the same length")
    predicted_pairs = set(pairs_from_extended_dot_bracket(predicted))
    reference_pairs = set(pairs_from_extended_dot_bracket(reference))
    denominator = len(predicted_pairs) + len(reference_pairs)
    if not denominator:
        return 0.0
    return 2.0 * len(predicted_pairs & reference_pairs) / denominator


def squared_mountain_distance(predicted: str, reference: str) -> float:
    """Return the sum of squared differences between profile heights."""
    if len(predicted) != len(reference):
        raise ValueError("Structures must have the same length")
    return float(
        sum(
            (predicted_height - reference_height) ** 2
            for predicted_height, reference_height in zip(
                mountain_heights(predicted), mountain_heights(reference)
            )
        )
    )


def mean_squared_mountain_distance(predicted: str, reference: str) -> float:
    """Return squared profile distance divided by the number of boundaries."""
    return squared_mountain_distance(predicted, reference) / max(len(reference) - 1, 1)


def normalized_squared_mountain_distance(predicted: str, reference: str) -> float:
    """Return squared profile distance normalized by its length-specific bound."""
    if len(predicted) != len(reference):
        raise ValueError("Structures must have the same length")
    n = len(reference)
    denominator = sum(min(boundary, n - boundary) ** 2 for boundary in range(1, n))
    if not denominator:
        return 0.0
    return squared_mountain_distance(predicted, reference) / denominator
