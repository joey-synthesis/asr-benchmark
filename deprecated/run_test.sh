#!/bin/bash
# Helper script to run the ASR test with the virtual environment

# Activate virtual environment
source .venv/bin/activate

# Run the test script
python test_asr.py

# Deactivate when done
deactivate
