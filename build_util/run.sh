#!/bin/bash
[ "$1" ] || { echo "please use options \"build\" or \"run\""; exit 1; }
if [ "$1" == "build" ]; then
    "$(dirname "$0")/build.py" print run=test.TrivialTest
else
    "$(dirname "$0")/build.py" /dontcompile run=test.TrivialTest
fi
