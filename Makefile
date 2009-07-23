
help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."

clean:
	mvn clean
	rm -f *timestamp */*timestamp
	rm -r ~/.m2/repository/edu/berkeley/cs/sketch
