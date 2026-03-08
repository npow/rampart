"""Testing primitives for Rampart: MockTool, cassette record/replay."""

from ._cassette import cassette
from ._mock_tools import MockTool

__all__ = [
    "MockTool",
    "cassette",
]
