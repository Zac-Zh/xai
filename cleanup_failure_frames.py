#!/usr/bin/env python3
"""
Cleanup script for failure_frames directories

This script helps free up disk space by removing failure_frames directories
that were generated with the old unoptimized code (150 frames × 900KB each).

Usage:
    python cleanup_failure_frames.py --dry-run          # See what would be deleted
    python cleanup_failure_frames.py --confirm          # Actually delete files
"""

import os
import shutil
import argparse
from pathlib import Path


def get_directory_size(path):
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_directory_size(entry.path)
    except PermissionError:
        pass
    return total


def format_size(bytes_size):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def find_failure_frames_dirs(root_dir="."):
    """Find all failure_frames directories."""
    failure_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if "failure_frames" in dirs:
            failure_frames_path = os.path.join(root, "failure_frames")
            failure_dirs.append(failure_frames_path)
    return failure_dirs


def cleanup_failure_frames(dry_run=True):
    """Clean up failure_frames directories."""
    print("Searching for failure_frames directories...")
    failure_dirs = find_failure_frames_dirs()

    if not failure_dirs:
        print("No failure_frames directories found.")
        return

    print(f"\nFound {len(failure_dirs)} failure_frames director{'y' if len(failure_dirs) == 1 else 'ies'}:")

    total_size = 0
    for dir_path in failure_dirs:
        size = get_directory_size(dir_path)
        total_size += size
        num_subdirs = len([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
        print(f"  {dir_path}")
        print(f"    Size: {format_size(size)}")
        print(f"    Subdirectories: {num_subdirs}")

    print(f"\nTotal size to be freed: {format_size(total_size)}")

    if dry_run:
        print("\n[DRY RUN] No files were deleted. Use --confirm to actually delete.")
        return

    print("\n⚠️  WARNING: This will permanently delete all failure_frames directories!")
    response = input("Type 'DELETE' to confirm: ")

    if response != "DELETE":
        print("Cancelled.")
        return

    print("\nDeleting failure_frames directories...")
    for dir_path in failure_dirs:
        try:
            shutil.rmtree(dir_path)
            print(f"  ✓ Deleted: {dir_path}")
        except Exception as e:
            print(f"  ✗ Failed to delete {dir_path}: {e}")

    print(f"\n✓ Cleanup complete! Freed approximately {format_size(total_size)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup failure_frames directories to free disk space"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    group.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete the directories"
    )

    args = parser.parse_args()

    cleanup_failure_frames(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
