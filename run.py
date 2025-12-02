"""
Wrapper script to run the Person Detection and Counting System.

This script allows running the application from the project root directory.
"""

import sys
from src.person_counter import main

if __name__ == "__main__":
    sys.exit(main())
