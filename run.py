# DO NOT MODIFY THIS FILE
# This is the entry point for your submission.
# Changing this file will probably fail your submissions.

import train
import test_FindCave

import os

# By default, only do testing
EVALUATION_STAGE = os.getenv('EVALUATION_STAGE', 'testing')

# Testing Phase
if EVALUATION_STAGE in ['all', 'testing']:
    test_FindCave.main()

