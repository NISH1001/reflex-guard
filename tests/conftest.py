import pytest
from loguru import logger


@pytest.fixture
def logs():
    """Messages loguru emits during the test."""
    out = []
    handler = logger.add(lambda m: out.append(m.record["message"]), level="DEBUG")
    yield out
    logger.remove(handler)
