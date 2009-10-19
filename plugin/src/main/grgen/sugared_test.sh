#!/bin/bash
grgen_dir="$(dirname "$0")"
python "$grgen_dir"/transform_sketch.py --gxl_file=/home/gatoatigrado/sandbox/skalch/base/src/test/scala/angelic/simple/SugaredTest.scala.ast.gxl "$@"
