#!/usr/bin/env python3
"""Cleanup utility for test sessions."""

import shutil
from pathlib import Path


def cleanup_test_sessions(storage_dir: Path = Path("sessions_test")):
    """
    Remove test session directory.

    Args:
        storage_dir: Path to test sessions directory
    """
    if storage_dir.exists():
        print(f"Removing {storage_dir}...")
        shutil.rmtree(storage_dir)
        print(f"✓ Cleaned up {storage_dir}")
    else:
        print(f"No test sessions found at {storage_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup test sessions")
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("sessions_test"),
        help="Test sessions directory to remove (default: sessions_test)"
    )

    args = parser.parse_args()
    cleanup_test_sessions(args.storage_dir)
