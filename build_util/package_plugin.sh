#!/bin/bash
skalch_proj_dir="$(dirname "$(dirname "$0")")"
cd "$skalch_proj_dir"
set -v
build_util/build.py fsc /plugin print
cp src/skalch/plugins/scalac-plugin.xml bin/skalch/plugins/
(cd bin; jar c0f ../lib/sketchrewriter.jar skalch/plugins/*)
