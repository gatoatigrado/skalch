
help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."

clean:
	mvn clean
	rm -f *timestamp */*timestamp
	rm -r ~/.m2/repository/edu/berkeley/cs/sketch

bitonic:
	cd skalch-base; mvn exec:java -Dexec.classpathScope=test -Dexec.mainClass=test.BitonicSortTest -Dexec.args="--array_length 4 --num_steps 10"
