#!/bin/bash
# Test script for docking regressor

set -e

echo "Running docking regressor tests..."

# Run Python tests
python3 -m pytest tests/test_docking_regressor.py -v

echo "All tests passed!"

