
help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(#.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" Makefile | sed -r "s/^# /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

clean:
	mvn clean
	rm -f *timestamp */*timestamp
	rm -r ~/.m2/repository/edu/berkeley/cs/sketch

# pom management utilities

set_version: # args: current=<version> next=<version>
	( [ "$(current)" ] && [ "$(next)" ] ) || { echo "please set current and next."; exit 1; }
	sed -i "s/$(current)/$(next)/g" pom.xml
	sed -i "s/$(current)/$(next)/g" skalch-plugin/pom.xml
	sed -i "s/$(current)/$(next)/g" skalch-base/pom.xml

kate: # open various config files in Kate
	kate -u Makefile pom.xml */pom.xml */db/*.xml

# other

bitonic_plugin: # build the plugin and compile the bitonic sort test
	cd skalch-plugin; mvn compile install
	cd skalch-base; export TRANSLATE_SKETCH=true; touch src/test/BitonicSortTest.scala; mvn test-compile -Dmaven.scala.displayCmd=true

plugin_dev: # build the plugin and compile the a test given by $(testfile)
	cd skalch-plugin; mvn compile install
	cd skalch-base; export TRANSLATE_SKETCH=true; touch src/test/$(testfile); mvn test-compile -Dmaven.scala.displayCmd=true

# tests

bitonic_test: # run the bitonic sort test
	cd skalch-base; mvn exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.BitonicSortTest -Dexec.args="--array_length 4 --num_steps 10"

completed_test: # run the completed test
	cd skalch-base; mvn compile test-compile exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.CompletedTest -Dexec.args=""

# developer-specific commands

gatoatigrado-clean-other: clean # clean relative paths in gatoatigrado's project
	rm -rf ../sketch/target ../sketch/mvn-bin ../sketch-util/target

gatoatigrado-build-plugin-deps: # build dependencies for the plugin
	# maven is messed up, or maybe this is Eclipse's build system
	rm -rf ~/sandbox/eclipse/sketch/target/classes/sketch/util
	cd ../sketch-util; mvn install
	cd ../sketch; mvn install

gatoatigrado-plugin-dev: gatoatigrado-build-plugin-deps # gatoatigrado's plugin development (bitonic sort sketch)
	@make bitonic_plugin

gatoatigrado-plugin-rbtree: gatoatigrado-build-plugin-deps # gatoatigrado's plugin development trying the red-black tree sketch
	@make plugin_dev testfile=RedBlackTreeTest.scala

gatoatigrado-plugin-dws: gatoatigrado-build-plugin-deps # dws sketch (lots of syntax)
	@make plugin_dev testfile=Dfs.scala

gatoatigrado-plugin-roman-numeral: gatoatigrado-build-plugin-deps # roman numerals (match stmt)
	@make plugin_dev testfile=RomanNumerals.scala

gatoatigrado-plugin-rev-list: gatoatigrado-build-plugin-deps # rev list test (catch stmt)
	@make plugin_dev testfile=RevListTest.scala

gatoatigrado: gatoatigrado-plugin-dev # whatever gatoatigrado's currently working on
