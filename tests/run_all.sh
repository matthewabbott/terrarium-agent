#!/bin/bash
# Test runner for Terrarium Agent

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Terrarium Agent Test Suite"
echo "========================================"
echo

# Activate venv
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠️  venv not found, using system Python"
fi

echo

# Check vLLM server
echo "Checking vLLM server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ vLLM server is running"
else
    echo "❌ vLLM server not responding"
    echo "Start it with: ./start_vllm_docker.sh"
    exit 1
fi

echo

# Change to project root
cd "$PROJECT_ROOT"

# Track results
PASSED=0
FAILED=0

# Run each test
for test_file in tests/test_*.py; do
    if [ -f "$test_file" ]; then
        test_name=$(basename "$test_file")
        echo "Running $test_name..."

        if python "$test_file"; then
            echo "✓ $test_name passed"
            ((PASSED++))
        else
            echo "❌ $test_name failed"
            ((FAILED++))
        fi

        echo
    fi
done

# Summary
echo "========================================"
echo "Test Results"
echo "========================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo

# Cleanup
echo "Cleaning up test sessions..."
python -m tests.cleanup
echo

# Exit code
if [ $FAILED -gt 0 ]; then
    echo "❌ Some tests failed"
    exit 1
else
    echo "✓ All tests passed!"
    exit 0
fi
