# @code standards ignore file

SHELL=/bin/bash

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -r "s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

clean:
	zsh -c "setopt -G; rm -f **/*timestamp **/*pyc **/*~ **/skalch/plugins/type_graph.gxl"
	zsh -c "setopt -G; rm -rf **/(bin|target) .gen **/gen/"

clean-gxl:
	zsh -c "setopt -G; rm -rf base/target/test-classes base/**/*.gxl"

test: killall
	echo "TODO -- run mvn test when it works again."
	(cd plugin/src/test/grgen; make test)
	@echo "TEST SUCCEEDED"

### utilities

cs: # code standards check, ignoring skalch_old, etc/util, no license, and Java files (eclipse formatter reigns supreme)
	build_util/code_standards.py | sed -r "/skalch_old/d; /ec\/util/d; /no license/d; /long line.+\.java/d"

kate: # open various config files in Kate
	kate -u Makefile pom.xml */pom.xml

remove-whitespace: # trim trailing whitespace on all files
	bash -c "source build_util/bash_functions.sh; cd plugin; trim_whitespace src"
	bash -c "source build_util/bash_functions.sh; cd base; trim_whitespace src"

install-plugin: gen
	(cd plugin; mvn assembly:assembly install)

compile: install-plugin
	mvn compile test-compile

py-fsc-compile:
	python scripts/compile_all.py

### Compile various tests using the plugin (to test the plugin)

grgen-devel:
	python scripts/compile_all.py
	python scripts/rulegen/rulegen.py

gxltosketch-devel: clean-gxl gen py-fsc-compile grgen java_gxlimport

killall:
	@killall mono 2>/dev/null; true

gen:
	base/src/codegen/generate_files.py plugin/src/main/grgen/ScalaAstModel.gm.jinja2
	cd plugin/src/main/grgen; grshell -N generate_typegraph.grs
	base/src/codegen/generate_files.py --no_rebuild

sugared_plugin_gxl: install-plugin
	@make plugin_dev testfile=angelic/simple/SugaredTest.scala

### grgen commands

new-unified-module:
	cd plugin/src/main/grgen; zsh "new_unified.zsh"

grgen: gen killall
	plugin/src/main/grgen/sugared_test.sh; make killall

ycomp: gen killall
	
	plugin/src/main/grgen/sugared_test.sh --ycomp --runonly; make killall

ycomp-intermediate: killall
	python plugin/src/main/grgen/transform_sketch.py --gxl_file=base/src/test/scala/angelic/simple/SugaredTest.intermediate.ast.gxl --grs_template="!/ycomp_intermediate.grs"

ycomp-unprocessed: killall
	python plugin/src/main/grgen/transform_sketch.py --ycomp "--gxl_file=base/src/test/scala/angelic/simple/SugaredTest.scala.ast.gxl" --grs_template='!/ycomp_intermediate.grs'

plugin_dev: # build the plugin and compile the a test given by $(testfile)
	cd base; export TRANSLATE_SKETCH=true; touch src/test/scala/$(testfile) && mvn compile test-compile -Dmaven.scala.displayCmd=true

java_gxlimport: gen
	(cd base; mvn -e compile exec:java "-Dexec.mainClass=sketch.compiler.parser.gxlimport.GxlImport" "-Dexec.args=src/test/scala/angelic/simple/SugaredTest.intermediate.ast.gxl")



### Sketch tests; use EXEC_ARGS=args to pass arguments

angelic_sketch: plugin_angelic_sketch # new angelic sketch base
	cd base; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=angelic.simple.SugaredTest" -Dexec.args="$(EXEC_ARGS)"

run_test: plugin_dev # run TEST_CLASS=<canonical java class name>	
	cd base; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=$(TEST_CLASS)" "-Dexec.args=$(EXEC_ARGS)"



### developer-specific commands

gatoatigrado-mv-grgen-files-todirectory:
	mkdir -p tmp-grgen
	gxlname=$$(grep import ~/.sketch/tmp/transform.grs | sed -r 's/.+"([^"]+)".+/\1/g'); \
            dirname="$$(dirname "$$gxlname")"; \
            cat ~/.sketch/tmp/transform.grs | sed "s/$${dirname//\//\\/}\///g" > tmp-grgen/transform.grs; \
            cp "$$gxlname" tmp-grgen
	cp -r plugin/src/main/grgen/{AllRules.grg,ScalaAstModel.gm,rules,stages-scripts} tmp-grgen
