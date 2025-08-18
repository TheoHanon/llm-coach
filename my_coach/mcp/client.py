import asyncio
import json

from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool

# MCP SDK imports
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client



class MCPTool(BaseTool):

    cmd : str
    mcp_args : Optional[list[str]] = None
    env : Optional[Dict[str, str]] = None
    mcp_tool_name : str
    timeout_s: Optional[float] = 30.0


    async def _arun(self, **kwargs) -> str:
        server_params = StdioServerParameters(
                command = self.cmd,
                args = self.mcp_args or [],
                env = self.env,
            )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                call = session.call_tool(self.mcp_tool_name, arguments=kwargs)
                result = await asyncio.wait_for(call, timeout=self.timeout_s)



        if getattr(result, "structuredContent", None) is not None:
            return json.dumps(result.structuredContent, ensure_ascii=False)
        if result.content:
            block = result.content[0]
            if isinstance(block, types.TextContent):
                return block.text
        return ""

    def _run(
        self, **kwargs   
    ) -> str :

        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # no running loop → fine to use asyncio.run
            return asyncio.run(self._arun(**kwargs))
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: asyncio.run(self._arun(**kwargs)))
                return fut.result()


















