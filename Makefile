# @code standards ignore file

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" Makefile | sed -r "s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

clean:
	mvn clean
	rm -rf bin target */{bin,target} ~/.m2/repository/edu/berkeley/cs/sketch *timestamp */*timestamp

### utilities

cs: # code standards check, ignoring skalch_old, etc/util, no license, and Java files (eclipse formatter reigns supreme)
	build_util/code_standards.py | sed -r "/skalch_old/d; /ec\/util/d; /no license/d; /long line.+\.java/d"

set_version: # args: current=<version> next=<version>
	( [ "$(current)" ] && [ "$(next)" ] ) || { echo "please set current and next."; exit 1; }
	sed -i "s/$(current)/$(next)/g" pom.xml
	sed -i "s/$(current)/$(next)/g" skalch-plugin/pom.xml
	sed -i "s/$(current)/$(next)/g" skalch-base/pom.xml

kate: # open various config files in Kate
	kate -u Makefile pom.xml */pom.xml

remove-whitespace: # trim trailing whitespace on all files
	bash -c "source build_util/bash_functions.sh; cd skalch-plugin; trim_whitespace src"
	bash -c "source build_util/bash_functions.sh; cd skalch-base; trim_whitespace src"

set-test-package-decls: # hack to use sed and rename all java package declarations in the tests directory
	bash -c "source build_util/bash_functions.sh; \
                set_package_decls skalch-base/src/test/scala; \
                set_package_decls skalch-base/src/main/java"



### Compile various tests using the plugin (to test the plugin)

plugin_sugared:
	@make plugin_dev testfile=skalch_old/simple/SugaredTest.scala

plugin_angelic_sketch:
	@make plugin_dev testfile=angelic/simple/SugaredTest.scala

plugin_dev: # build the plugin and compile the a test given by $(testfile)
	cd skalch-plugin; mvn -o install
	cd skalch-base; export TRANSLATE_SKETCH=true; touch src/test/scala/$(testfile); mvn -o test-compile -Dmaven.scala.displayCmd=true



### Sketch tests; use EXEC_ARGS=args to pass arguments

angelic_sketch: plugin_angelic_sketch # new angelic sketch base
	cd skalch-base; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=angelic.simple.SugaredTest" -Dexec.args="$(EXEC_ARGS)"

run_test: plugin_dev # run TEST_CLASS=<canonical java class name> EXEC_ARGS=<args>	
	cd skalch-base; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=$(TEST_CLASS)" "-Dexec.args=$(EXEC_ARGS)"

run_test_debug: plugin_dev # run debug TEST_CLASS=<canonical java class name> EXEC_ARGS=<args>	
	cd skalch-base; export MAVEN_OPTS="-Xdebug -Xrunjdwp:transport=dt_socket,address=8998,server=y,suspend=y"; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=$(TEST_CLASS)" "-Dexec.args=$(EXEC_ARGS)"




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
