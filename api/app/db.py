import asyncpg
from fastapi import Request


def get_pool(request: Request) -> asyncpg.Pool:
    """Return the application's shared asyncpg connection pool."""
    return request.app.state.pool