# @code standards ignore file

TRANSFORM_SCRIPT = ../../main/grgen/transform_sketch.py

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -r "s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

test_construct_assignable:
	python "$(TRANSFORM_SCRIPT)" --gxl_file=construct_assignable/test.gxl --grs_template=construct_assignable/test.grs

test_dont_delete:
	python "$(TRANSFORM_SCRIPT)" --gxl_file=dont_delete_nodes/test.gxl --grs_template=dont_delete_nodes/test.grs

test: test_construct_assignable test_dont_delete
