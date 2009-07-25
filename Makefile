
help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "\w+:( #.+)?$$" Makefile | sed "s/://g; s/^/    /g; s/#/\\n        /g"

clean:
	mvn clean
	rm -f *timestamp */*timestamp
	rm -r ~/.m2/repository/edu/berkeley/cs/sketch

set_version: # args: current=<version> next=<version>
	( [ "$(current)" ] && [ "$(next)" ] ) || { echo "please set current and next."; exit 1; }
	sed -i "s/$(current)/$(next)/g" pom.xml
	sed -i "s/$(current)/$(next)/g" skalch-plugin/pom.xml
	sed -i "s/$(current)/$(next)/g" skalch-base/pom.xml

bitonic_plugin: # build the plugin and compile the bitonic sort test
	cd skalch-plugin; mvn install
	cd skalch-base; export TRANSLATE_SKETCH=true; touch src/test/BitonicSortTest.scala; mvn test-compile

bitonic_test: # run the bitonic sort test
	cd skalch-base; mvn exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.BitonicSortTest -Dexec.args="--array_length 4 --num_steps 10"

completed_test: # run the completed test
	cd skalch-base; mvn compile test-compile exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.CompletedTest -Dexec.args=""
