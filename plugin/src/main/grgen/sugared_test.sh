#!/bin/bash
grgen_dir="$(dirname "$0")"
python "$grgen_dir"/transform_sketch.py --gxl_file=base/src/test/scala/angelic/simple/SugaredTest.scala.ast.gxl --output_file=base/src/test/scala/angelic/simple/SugaredTest.intermediate.ast.gxl "$@"
