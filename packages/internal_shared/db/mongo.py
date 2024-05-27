import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from internal_shared.logger import get_logger

_logger = get_logger(__name__)

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://db:27017")

_client = AsyncIOMotorClient(_MONGO_URI)
_sync_client = MongoClient(_MONGO_URI)


def get_sync_db(db_name: str):
    sync_db = _sync_client.get_database(db_name)

    try:
        sync_db.command("ping")
        _logger.info("MongoDB connected successfully to %s.", db_name)
    except ConnectionFailure:
        _logger.error("MongoDB connection failed to %s.", db_name, exc_info=True)
        return None

    return sync_db


async def get_async_db(db_name: str):
    async_db = _client.get_database(db_name)

    try:
        await async_db.command("ping")
        _logger.info("AsyncMongoDB connected successfully to %s.", db_name)
    except ConnectionFailure:
        _logger.error("AsyncMongoDB connection failed to %s.", db_name, exc_info=True)
        return None

    return async_db
