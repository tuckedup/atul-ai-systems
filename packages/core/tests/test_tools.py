import mcp
import pytest
from aisys.tools import Registry, Tool
from pydantic import BaseModel


class Args(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_registered_tool_mcp_round_trip():
    local = Registry()
    local.register(Tool("double", lambda value: value * 2, "low", "Double an integer", Args))
    async with mcp.Client(local.mcp_server()) as client:
        listed = await client.list_tools()
        assert any(item.name == "double" for item in listed.tools)
        result = await client.call_tool("double", {"value": 4})
    assert "8" in str(result)
