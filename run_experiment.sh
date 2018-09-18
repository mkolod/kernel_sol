#!/bin/bash

nvprof --profile-from-start off -f -o foo.nvvp -- python foo.py && python correlate.py foo.nvvp

