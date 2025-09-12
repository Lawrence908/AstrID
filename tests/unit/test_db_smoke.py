"""Minimal DB smoke tests using the shared async session fixture."""

from __future__ import annotations

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_can_execute_simple_select(db_session):
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
