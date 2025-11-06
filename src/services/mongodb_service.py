"""MongoDB service for chemistry compound data."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

load_dotenv(override=True)


class MongoDBService:
    """Service class for MongoDB operations."""

    def __init__(self):
        """Initialize MongoDB client."""
        self.uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGODB_DATABASE", "chemistry_db")
        self.compounds_collection_name = "compounds"
        self.rules_collection_name = "rules"

        self.client: MongoClient = MongoClient(self.uri)
        self.db: Database = self.client[self.db_name]
        self.compounds: Collection = self.db[self.compounds_collection_name]
        self.rules: Collection = self.db[self.rules_collection_name]

    def get_all_compounds(self) -> List[Dict[str, Any]]:
        """Get all compounds from MongoDB.

        Returns:
            List of compound documents
        """
        compounds = list(self.compounds.find({}, {"_id": 0}))
        return compounds

    def get_compound_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get compound by doc_id.

        Args:
            doc_id: Compound document ID

        Returns:
            Compound document or None if not found
        """
        return self.compounds.find_one({"doc_id": doc_id}, {"_id": 0})

    def get_compounds_by_class(self, chemical_class: str) -> List[Dict[str, Any]]:
        """Get compounds by chemical class.

        Args:
            chemical_class: Chemical class (e.g., "Ancol (Alcohol)")

        Returns:
            List of compound documents
        """
        compounds = list(self.compounds.find({"class": chemical_class}, {"_id": 0}))
        return compounds

    def insert_compound(self, compound: Dict[str, Any]) -> str:
        """Insert a single compound.

        Args:
            compound: Compound document

        Returns:
            Inserted document ID
        """
        result = self.compounds.insert_one(compound)
        return str(result.inserted_id)

    def insert_compounds(self, compounds: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple compounds.

        Args:
            compounds: List of compound documents

        Returns:
            List of inserted document IDs
        """
        result = self.compounds.insert_many(compounds)
        return [str(id) for id in result.inserted_ids]

    def update_compound(self, doc_id: str, update_data: Dict[str, Any]) -> bool:
        """Update compound by doc_id.

        Args:
            doc_id: Compound document ID
            update_data: Fields to update

        Returns:
            True if updated, False otherwise
        """
        result = self.compounds.update_one(
            {"doc_id": doc_id},
            {"$set": update_data}
        )
        return result.modified_count > 0

    def delete_compound(self, doc_id: str) -> bool:
        """Delete compound by doc_id.

        Args:
            doc_id: Compound document ID

        Returns:
            True if deleted, False otherwise
        """
        result = self.compounds.delete_one({"doc_id": doc_id})
        return result.deleted_count > 0

    def clear_compounds(self):
        """Delete all compounds."""
        self.compounds.delete_many({})

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all chemistry rules.

        Returns:
            List of rule documents
        """
        rules = list(self.rules.find({}, {"_id": 0}))
        return rules

    def insert_rule(self, rule: Dict[str, Any]) -> str:
        """Insert a chemistry rule.

        Args:
            rule: Rule document

        Returns:
            Inserted document ID
        """
        result = self.rules.insert_one(rule)
        return str(result.inserted_id)

    def clear_rules(self):
        """Delete all rules."""
        self.rules.delete_many({})

    def create_indexes(self):
        """Create indexes for better query performance."""
        # Compound indexes
        self.compounds.create_index("doc_id", unique=True)
        self.compounds.create_index("iupac_name")
        self.compounds.create_index("class")
        self.compounds.create_index("formula")

        # Text index for full-text search
        self.compounds.create_index([
            ("iupac_name", "text"),
            ("common_names", "text"),
            ("formula", "text"),
            ("info", "text")
        ])

        print("✅ Created MongoDB indexes")

    def close(self):
        """Close MongoDB connection."""
        self.client.close()


# Global instance
mongodb_service = MongoDBService()
