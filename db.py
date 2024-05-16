from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://db:27017/evaluation_db")

client = AsyncIOMotorClient(MONGO_URI)
db = client.evaluation_db

# Synchronous client for initial setup or admin tasks
sync_client = MongoClient(MONGO_URI)
sync_db = sync_client.get_database("evaluation_db")

# Test connection
try:
    sync_client.admin.command('ping')
    sync_db.command('ping')
    print("MongoDB connected successfully")
except ConnectionFailure:
    print("MongoDB connection failed")
