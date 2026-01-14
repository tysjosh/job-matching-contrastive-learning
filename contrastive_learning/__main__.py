"""
Entry point for running contrastive learning CLI as a module.

Usage:
    python -m contrastive_learning [command] [options]
"""

from .cli import main

if __name__ == '__main__':
    exit(main())