#!/usr/bin/env python3
"""
Migrate sessions from flat structure to date-based organization.

Old structure:
  sessions/{type}/{id}.json

New structure:
  sessions/{type}/{date}/{id}.json

The date is extracted from the created_at field in the session metadata.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


def migrate_session(session_file: Path, dry_run: bool = False) -> bool:
    """
    Migrate a single session file to date-based organization.

    Args:
        session_file: Path to session JSON file
        dry_run: If True, don't actually move files

    Returns:
        True if migration successful or not needed
    """
    try:
        # Read session data
        data = json.loads(session_file.read_text())
        metadata = data.get("metadata", {})

        # Extract date from created_at
        created_at = metadata.get("created_at")
        if not created_at:
            print(f"  ⚠️  No created_at in {session_file}, using current date")
            session_date = datetime.now().strftime("%Y-%m-%d")
        else:
            # Parse ISO format timestamp and extract date
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            session_date = dt.strftime("%Y-%m-%d")

        # Determine target path
        # session_file is: sessions/{type}/{id}.json
        # target is: sessions/{type}/{date}/{id}.json
        type_dir = session_file.parent
        date_dir = type_dir / session_date
        target_file = date_dir / session_file.name

        # Check if already in date-based structure
        if session_file.parent.parent.parent.name == "sessions":
            # File is already in a date subdirectory
            print(f"  ✓ Already migrated: {session_file}")
            return True

        # Check if target already exists
        if target_file.exists():
            print(f"  ⚠️  Target already exists: {target_file}")
            return False

        # Migrate
        print(f"  → Migrating: {session_file.name}")
        print(f"     to: {type_dir.name}/{session_date}/{session_file.name}")

        if not dry_run:
            # Create date directory
            date_dir.mkdir(parents=True, exist_ok=True)

            # Add session_date to metadata if not present
            if "session_date" not in metadata:
                metadata["session_date"] = session_date
                data["metadata"] = metadata

                # Write updated data to new location
                target_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

                # Remove old file
                session_file.unlink()
            else:
                # Just move the file
                shutil.move(str(session_file), str(target_file))

        return True

    except Exception as e:
        print(f"  ❌ Error migrating {session_file}: {e}")
        return False


def migrate_directory(storage_dir: Path = Path("sessions"), dry_run: bool = False):
    """
    Migrate all sessions in a storage directory.

    Args:
        storage_dir: Root sessions directory
        dry_run: If True, don't actually move files
    """
    print("=" * 60)
    print("Session Migration Utility")
    print("=" * 60)
    print(f"Storage directory: {storage_dir}")
    print(f"Dry run: {dry_run}")
    print()

    if not storage_dir.exists():
        print(f"❌ Directory not found: {storage_dir}")
        return

    migrated = 0
    skipped = 0
    failed = 0

    # Iterate through context type directories
    for type_dir in storage_dir.iterdir():
        if not type_dir.is_dir():
            continue

        context_type = type_dir.name
        print(f"Processing {context_type}/ ...")

        # Find all JSON files in the type directory (flat structure)
        flat_sessions = list(type_dir.glob("*.json"))

        if not flat_sessions:
            print(f"  No flat sessions found")
            print()
            continue

        print(f"  Found {len(flat_sessions)} flat session(s)")

        # Migrate each session
        for session_file in flat_sessions:
            success = migrate_session(session_file, dry_run)

            if success:
                migrated += 1
            else:
                failed += 1

        print()

    # Summary
    print("=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Migrated: {migrated}")
    print(f"Failed: {failed}")
    print()

    if dry_run:
        print("This was a DRY RUN - no files were moved")
        print("Run without --dry-run to perform actual migration")
    else:
        print("Migration complete!")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate sessions from flat to date-based organization"
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("sessions"),
        help="Root sessions directory (default: ./sessions)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually move files, just show what would happen"
    )

    args = parser.parse_args()

    migrate_directory(args.storage_dir, args.dry_run)
