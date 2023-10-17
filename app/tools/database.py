from pymongo import MongoClient


class DatabaseHandler:
    def __init__(self, db_url, db_name):
        self.client = MongoClient(db_url, serverSelectionTimeoutMS=10000, connect=False)
        self.db = self.client[db_name]

    def distinct(self, collection_name, field, query=None):
        collection = self.db[collection_name]
        return collection.distinct(field, query)

    def find(self, collection_name, query, projection=None, sort=None, limit=None):
        collection = self.db[collection_name]
        cursor = collection.find(query, projection)
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def find_one_and_update(self, collection_name, query, update_data):
        collection = self.db[collection_name]
        return collection.find_one_and_update(query, {"$set": update_data})

    def insert_one(self, collection_name, data):
        collection = self.db[collection_name]
        return collection.insert_one(data)
