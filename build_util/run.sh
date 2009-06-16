#!/bin/bash
[ "$1" ] || { echo "please use options \"build\" or \"run\""; exit 1; }
cmd="$1"; shift
if [ "$cmd" == "build" ]; then
    "$(dirname "$0")/build.py" print run_app=test.TrivialTest --run_opt_list "$@"
else
    "$(dirname "$0")/build.py" /dontcompile run_app=test.TrivialTest --run_opt_list "$@"
fi
