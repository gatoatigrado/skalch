
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

# tests

bitonic_test: # run the bitonic sort test
	cd skalch-base; mvn exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.BitonicSortTest -Dexec.args="--array_length 4 --num_steps 10"

completed_test: # run the completed test
	cd skalch-base; mvn compile test-compile exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.CompletedTest -Dexec.args=""

# developer-specific commands

gatoatigrado-clean-other: clean # clean relative paths in gatoatigrado's project
	rm -rf ../sketch/target ../sketch/mvn-bin ../sketch-util/target

gatoatigrado-plugin-dev: # gatoatigrado's plugin development
	# maven is messed up, or maybe this is Eclipse's build system
	rm -rf ~/sandbox/eclipse/sketch/target/classes/sketch/util
	cd ../sketch-util; mvn install
	cd ../sketch; mvn install
	@make bitonic_plugin

gatoatigrado: gatoatigrado-plugin-dev # whatever gatoatigrado's currently working on
