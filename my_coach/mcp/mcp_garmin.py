import os
from .client import MCPTool
from pydantic import BaseModel, Field
from datetime import date

class SnapshotArgs(BaseModel):
    from_date: date = Field(description="Start date (YYYY-MM-DD)")
    to_date:   date = Field(description="End date (YYYY-MM-DD)")


garmin_tool = MCPTool(
    name = "garmin", 
    description="Retrieve the data of the user", 
    cmd = "uvx",
    mcp_args = [
    "--with", "garth-mcp-server",
    "garth-mcp-server"],     
    mcp_tool_name="snapshot",
    args_schema=SnapshotArgs,
    env = {"GARTH_TOKEN" : os.environ["GARTH_TOKEN"]},
)

