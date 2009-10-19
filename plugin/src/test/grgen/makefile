# @code standards ignore file

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -r "s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

test:
	python ../../main/grgen/transform_sketch.py --gxl_file=construct_assignable/test.gxl --grs_template=construct_assignable/test.grs
