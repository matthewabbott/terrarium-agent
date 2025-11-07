# Tests

Test suite for Terrarium Agent.

## Running Tests

### Run all tests:
```bash
./tests/run_all.sh
```

### Run individual test:
```bash
source venv/bin/activate
python tests/test_multi_context.py
python tests/test_persistence.py
```

### Cleanup test sessions:
```bash
python -m tests.cleanup
```

## Tests

- **test_multi_context.py** - Tests multi-context management, context swapping, and LRU cache
- **test_persistence.py** - Tests session persistence across "restarts"

## Prerequisites

- vLLM server running (`./start_vllm_docker.sh`)
- Python virtual environment activated
- Dependencies installed (`pip install -r requirements.txt`)

## Test Sessions

Tests use `sessions_test/` directory for test data. This directory is:
- Automatically created during tests
- Cleaned up by `run_all.sh` after tests complete
- Ignored by git (`.gitignore`)
- Can be manually cleaned with `python -m tests.cleanup`

## Adding New Tests

1. Create `tests/test_your_feature.py`
2. Follow the naming convention: `test_*.py`
3. Make it executable: `chmod +x tests/test_your_feature.py`
4. It will be automatically picked up by `run_all.sh`

Example test structure:
```python
#!/usr/bin/env python3
"""Test your feature."""

import asyncio

async def test_your_feature():
    """Test description."""
    # Your test code
    assert True
    print("✓ Test passed")

if __name__ == "__main__":
    asyncio.run(test_your_feature())
```

## CI/CD

To integrate with CI/CD:
```bash
# In CI script
./start_vllm_docker.sh &
sleep 60  # Wait for model to load
./tests/run_all.sh
```
