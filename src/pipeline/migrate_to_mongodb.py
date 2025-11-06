"""Migrate compound data from JSON file to MongoDB."""

import json
from pathlib import Path
from ..services.mongodb_service import mongodb_service


def migrate_compounds():
    """Load compounds from single JSON file and insert into MongoDB."""
    compounds_file = Path(__file__).parent.parent / "data" / "compounds.json"

    print("🔄 Loading compounds from JSON file...")
    with open(compounds_file, "r", encoding="utf-8") as f:
        compounds = json.load(f)

    print(f"📊 Loaded {len(compounds)} compounds")
    for compound in compounds:
        print(f"  ✓ {compound['doc_id']} ({compound['iupac_name']})")

    # Clear existing data
    print("\n🗑️  Clearing existing compounds...")
    mongodb_service.clear_compounds()

    # Insert all compounds
    print("\n💾 Inserting compounds into MongoDB...")
    mongodb_service.insert_compounds(compounds)

    print(f"✅ Successfully inserted {len(compounds)} compounds")

    # Verify
    count = mongodb_service.compounds.count_documents({})
    print(f"\n✓ Verification: {count} compounds in MongoDB")


def migrate_rules():
    """Load rules from single JSON file and insert into MongoDB."""
    rules_file = Path(__file__).parent.parent / "data" / "rules.json"

    if not rules_file.exists():
        print("\n⚠️  No rules.json file found, skipping...")
        return

    print("\n🔄 Loading rules from JSON file...")
    with open(rules_file, "r", encoding="utf-8") as f:
        rules = json.load(f)

    if not rules:
        print("  ⚠️  No rules found in file")
        return

    print(f"📊 Loaded {len(rules)} rules")
    for rule in rules:
        rule_name = rule.get('rule_id', rule.get('name', 'Unknown'))
        print(f"  ✓ {rule_name}")

    # Clear existing rules
    print("\n🗑️  Clearing existing rules...")
    mongodb_service.clear_rules()

    # Insert all rules
    print("\n💾 Inserting rules into MongoDB...")
    for rule in rules:
        mongodb_service.insert_rule(rule)

    print(f"✅ Successfully inserted {len(rules)} rules")

    # Verify
    count = mongodb_service.rules.count_documents({})
    print(f"\n✓ Verification: {count} rules in MongoDB")


def create_indexes():
    """Create database indexes."""
    print("\n🔍 Creating indexes...")
    mongodb_service.create_indexes()


def main():
    """Main migration workflow."""
    print("=" * 60)
    print("  MongoDB Migration Script")
    print("  Chemistry Compound Data → MongoDB")
    print("=" * 60)

    try:
        # Step 1: Migrate compounds
        migrate_compounds()

        # Step 2: Migrate rules
        migrate_rules()

        # Step 3: Create indexes
        create_indexes()

        print("\n" + "=" * 60)
        print("  ✅ Migration Complete!")
        print("=" * 60)

        # Summary
        compound_count = mongodb_service.compounds.count_documents({})
        rule_count = mongodb_service.rules.count_documents({})

        print(f"\n📊 Database Summary:")
        print(f"  • Compounds: {compound_count}")
        print(f"  • Rules: {rule_count}")
        print(f"  • Database: {mongodb_service.db_name}")
        print(f"  • URI: {mongodb_service.uri}")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        mongodb_service.close()


if __name__ == "__main__":
    main()
