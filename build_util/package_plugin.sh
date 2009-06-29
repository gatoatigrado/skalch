#!/bin/bash
skalch_proj_dir="$(dirname "$(dirname "$0")")"
cd "$skalch_proj_dir"
set -v
rm -f lib/sketchrewriter.jar
rm -rf bin/skalch/plugins
build_util/build.py fsc no_default_opts --option=-classpath "--option=%(classpath)s" --option=-sourcepath "--option=%(src_path)s" /plugin print || exit 1
cp src/skalch/plugins/scalac-plugin.xml bin/
(cd bin; jar c0f ../lib/sketchrewriter.jar scalac-plugin.xml skalch/plugins/*)
fsc -shutdown
