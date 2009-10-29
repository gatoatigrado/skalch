# @code standards ignore file

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -r "s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

clean:
	mvn clean
	rm -rf bin target */{bin,target} ~/.m2/repository/edu/berkeley/cs/sketch *timestamp */*timestamp

test:
	echo "TODO -- run mvn test when it works again."
	(cd plugin/src/test/grgen; make test)
	@echo "TEST SUCCEEDED"

### utilities

cs: # code standards check, ignoring skalch_old, etc/util, no license, and Java files (eclipse formatter reigns supreme)
	build_util/code_standards.py | sed -r "/skalch_old/d; /ec\/util/d; /no license/d; /long line.+\.java/d"

set_version: # args: current=<version> next=<version>
	( [ "$(current)" ] && [ "$(next)" ] ) || { echo "please set current and next."; exit 1; }
	sed -i "s/$(current)/$(next)/g" pom.xml
	sed -i "s/$(current)/$(next)/g" plugin/pom.xml
	sed -i "s/$(current)/$(next)/g" base/pom.xml

kate: # open various config files in Kate
	kate -u Makefile pom.xml */pom.xml

remove-whitespace: # trim trailing whitespace on all files
	bash -c "source build_util/bash_functions.sh; cd plugin; trim_whitespace src"
	bash -c "source build_util/bash_functions.sh; cd base; trim_whitespace src"

set-test-package-decls: # hack to use sed and rename all java package declarations in the tests directory
	bash -c "source build_util/bash_functions.sh; \
                set_package_decls base/src/test/scala; \
                set_package_decls base/src/main/java"



### Compile various tests using the plugin (to test the plugin)

generate_type_graph:
	cd plugin/src/main/grgen; grshell generate_typegraph.grs

compile_install_plugin:
	cd plugin; mvn install

plugin_sugared:
	@make plugin_dev testfile=skalch_old/simple/SugaredTest.scala

plugin_angelic_sketch:
	@make plugin_dev testfile=angelic/simple/SugaredTest.scala

grgen_compile:
	plugin/src/main/grgen/sugared_test.sh

ycomp:
	@killall mono java 2>/dev/null; true
	plugin/src/main/grgen/sugared_test.sh --ycomp
	@killall mono java 2>/dev/null; true

jython_example:
	mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=org.python.util.jython" -Dexec.args="base/src/main/jython/print_graph.py base/src/test/scala/angelic/simple/SugaredTest.intermediate.ast.gxl"

plugin_dev: compile_install_plugin # build the plugin and compile the a test given by $(testfile)
	cd base; export TRANSLATE_SKETCH=true; touch src/test/scala/$(testfile); mvn test-compile -Dmaven.scala.displayCmd=true



### Sketch tests; use EXEC_ARGS=args to pass arguments

angelic_sketch: plugin_angelic_sketch # new angelic sketch base
	cd base; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=angelic.simple.SugaredTest" -Dexec.args="$(EXEC_ARGS)"

run_test: plugin_dev # run TEST_CLASS=<canonical java class name>	
	cd base; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=$(TEST_CLASS)" "-Dexec.args=$(EXEC_ARGS)"



### developer-specific commands

buildutil/build_info.pickle:
	buildutil/get_sources.zsh

fsc-plugin: buildutil/build_info.pickle
	buildutil/run_fsc.py --compile_plugin

fsc-base: buildutil/build_info.pickle
	buildutil/run_fsc.py --compile_base

fsc-test: buildutil/build_info.pickle
	buildutil/run_fsc.py --compile_test

gatoatigrado-build-plugin-deps: # build dependencies for the plugin, use skipdeps=1 to skip
ifndef skipdeps
	cd ../sketch-frontend; make g_inst
endif

# gatoatigrado's plugin development trying the red-black tree sketch
gatoatigrado-plugin-rbtree: gatoatigrado-build-plugin-deps
	@make plugin_dev testfile=RedBlackTreeTest.scala

# dws sketch (lots of syntax)
gatoatigrado-plugin-dws: gatoatigrado-build-plugin-deps
	@make plugin_dev testfile=Dfs.scala

# roman numerals (match stmt)
gatoatigrado-plugin-roman-numeral: gatoatigrado-build-plugin-deps
	@make plugin_dev testfile=RomanNumerals.scala

# rev list test (catch stmt)
gatoatigrado-plugin-rev-list: gatoatigrado-build-plugin-deps
	@make plugin_dev testfile=RevListTest.scala

g: gatoatigrado-build-plugin-deps plugin_angelic_sketch # whatever gatoatigrado's currently working on
