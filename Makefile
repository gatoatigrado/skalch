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

install-plugin:
	(cd skalch-plugin; mvn install)

compile: install-plugin
	mvn compile test-compile

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
	cd skalch-plugin; mvn install
	cd skalch-base; export TRANSLATE_SKETCH=true; touch src/test/scala/$(testfile); mvn test-compile -Dmaven.scala.displayCmd=true



### Sketch tests; use EXEC_ARGS=args to pass arguments

run_test: plugin_dev # run TEST_CLASS=<canonical java class name> EXEC_ARGS=<args>	
	cd skalch-base; export MAVEN_OPTS="-Djava.library.path=/home/sbarman/Research/workspace/angels/lib/"; mvn  -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=$(TEST_CLASS)" "-Dexec.args=$(EXEC_ARGS)"

run_test_debug: plugin_dev # run debug TEST_CLASS=<canonical java class name> EXEC_ARGS=<args>	
	cd skalch-base; export MAVEN_OPTS="-Xdebug -Xrunjdwp:transport=dt_socket,address=8998,server=y,suspend=y -Djava.library.path=/home/sbarman/Research/workspace/angels/lib/"; mvn -e exec:java "-Dexec.classpathScope=test" "-Dexec.mainClass=$(TEST_CLASS)" "-Dexec.args=$(EXEC_ARGS)"

