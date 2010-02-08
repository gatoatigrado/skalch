# @code standards ignore file

SHELL = /bin/bash

VERSION = 1.4.0

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -u -r "/ #HIDDEN/d; s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

help-all: # show uncommon commands as well
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -u -r "s/ #HIDDEN//g; s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

run: # run the gui
	python -m rulegen.rulegen
