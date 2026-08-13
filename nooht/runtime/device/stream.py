import asyncio
from enum import Enum
class StreamType(Enum): COMPUTE="compute"; LOAD="load"; SYNC="sync"
class AsyncStream:
    def __init__(self, stream_type: StreamType, max_concurrency: int = 1): self._sem = asyncio.Semaphore(max_concurrency)
    async def submit(self, coro):
        async with self._sem: return await coro
