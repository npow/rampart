"""Shared fixtures for Rampart tests."""

import pytest

from rampart.checkpointers import MemoryCheckpointer


@pytest.fixture
def mem_cp():
    """Fresh MemoryCheckpointer per test."""
    return MemoryCheckpointer()
