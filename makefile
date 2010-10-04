# @code standards ignore file

SHELL=/usr/bin/zsh
grgenfiles=$(shell source ./env && echo $$grgenfiles)
libgrg=$(shell source ./env && echo $$libgrg)
stagetf_sources=$(shell source ./env && echo $$stagetf_sources)

help:
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -u -r "/ #HIDDEN/d; s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

help-all: # show uncommon commands as well
	@echo "NOTE - this makefile is mostly unix aliases. Use 'mvn install' to build."
	@grep -iE "^(###.+|[a-zA-Z0-9_-]+:.*(#.+)?)$$" makefile | sed -u -r "s/ #HIDDEN//g; s/^### /\n/g; s/:.+#/#/g; s/^/    /g; s/#/\\n        /g; s/:$$//g"

clean:
	fsc -shutdown || true
	zsh -c "setopt -G; rm -rf base/target/test-classes base/**/*.gxl"
	zsh -c "setopt -G; rm -f **/*timestamp **/*.(pyc|dll|exe|pidb|mdb) **/*~ **/skalch/plugins/type_graph.gxl"
	zsh -c "setopt -G; rm -rf **/(bin|target) .gen **/gen/"
	zsh -c "setopt -G; rm -rf ~/.m2/repository/edu/berkeley/cs/sketch/skalch-plugin"

show-env-vars:
	@echo "\$$grgenfiles from $(origin grgenfiles), value '$(grgenfiles)'"
	@echo "\$$libgrg from $(origin libgrg), value '$(libgrg)'"
	@echo "\$$stagetf_sources from $(origin stagetf_sources), value '$(stagetf_sources)'"

test: gen stagetf-compile $(grgenfiles)/runtests.exe
	fsc -shutdown || true
	source env; mono $(grgenfiles)/runtests.exe "$$angelicsimple"/Test0003_WhileLoops.scala "$$angelicsimple"/Test0004_Classes.scala "$$angelicsimple"/Test0005_tprint.scala > test-output.txt 2>&1 || { grep -E "\\[ERROR\\]" test-output.txt; false; }



### open documentation in openoffice

swdoc:
	oowriter doc/software_documentation.odt

tutorial:
	oowriter doc/tutorial.odt

devdoc:
	lyx doc/developing.lyx



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

compile: install-plugin stagetf-compile
	mvn compile test-compile
	source env; tf --goal create_templates "$$rewritetemplates"

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
	@mkdir -p plugin/src/main/resources/skalch/plugins
	for i in $$grgenfiles/**/*.jinja2; do base/src/codegen/generate_files.py $$i; done
	for i in base/src/main/scala/skalch/cuda/**/*.jinja2; do base/src/codegen/generate_files.py $$i; done
	@make --quiet plugin/src/main/resources/skalch/plugins/type_graph.gxl
	base/src/codegen/generate_files.py --no_rebuild

plugin/src/main/resources/skalch/plugins/type_graph.gxl: plugin/src/main/grgen/ScalaAstModel.gm
	cd plugin/src/main/grgen; grshell -N generate_typegraph.grs

sugared_plugin_gxl: install-plugin
	@make plugin_dev testfile=angelic/simple/SugaredTest.scala



### grgen commands

new-unified-module:
	cd plugin/src/main/grgen; zsh "new_unified.zsh"

grgen: gen killall stagetf-compile
	source ./env; tf-test

stagetf-compile: $(libgrg)/fsharp_stage_transformer.dll $(grgenfiles)/transformer.exe # compile the stage transformer (computes changes in transformer.fs and its library)

$(grgenfiles)/transformer.exe: $(grgenfiles)/rewrite_rules.fs $(grgenfiles)/rewrite_stage_info.fs $(grgenfiles)/transformer.fs
	fsharpc --optimize+ -r:"$$libgrg"/fsharp_stage_transformer.dll -r:"$$libgrg"/GrIO.dll -r:"$$libgrg"/GrShell.dll -r:"$$libgrg"/lgspBackend.dll -r:"$$libgrg"/libGr.dll "--out:$@" $^

$(grgenfiles)/runtests.exe: $(grgenfiles)/transformer.exe $(grgenfiles)/runtests.fs
	fsharpc --optimize+ -r:"$$libgrg"/fsharp_stage_transformer.dll -r:"$$libgrg"/GrIO.dll -r:"$$libgrg"/GrShell.dll -r:"$$libgrg"/lgspBackend.dll -r:"$$libgrg"/libGr.dll -r:$(grgenfiles)/transformer.exe "--out:$@" $(grgenfiles)/runtests.fs

$(libgrg)/fsharp_stage_transformer.dll: $(stagetf_sources)
	@echo -e "\n\nNOTE -- compiling stage transformer library..."
	cd $(libgrg)/../FSharpBindings; ./build.zsh

ycomp: gen killall stagetf-compile
	source ./env; tf-test --debugbefore SketchFinalMinorCleanup

ycomp-intermediate: killall
	python plugin/src/main/grgen/transform_sketch.py --ycomp --gxl_file=base/src/test/scala/angelic/simple/SugaredTest.intermediate.ast.gxl --grs_template="!/ycomp_intermediate.grs"

ycomp-unprocessed: killall
	python plugin/src/main/grgen/transform_sketch.py --ycomp "--gxl_file=base/src/test/scala/angelic/simple/SugaredTest.scala.ast.gxl" --grs_template='!/ycomp_intermediate.grs'

plugin_dev: # build the plugin and compile the a test given by $(testfile)
	cd base; export TRANSLATE_SKETCH=true; touch src/test/scala/$(testfile) && mvn compile test-compile -Dmaven.scala.displayCmd=true

java_gxlimport: gen
	(cd base; mvn -e compile exec:java "-Dexec.mainClass=sketch.compiler.parser.gxlimport.GxlImport" "-Dexec.args=src/test/scala/angelic/simple/SugaredTest.sketch.ast.gxl")



### developer-specific commands

gatoatigrado-mv-grgen-files-todirectory:
	mkdir -p tmp-grgen
	gxlname=$$(grep import ~/.sketch/tmp/transform.grs | sed -r 's/.+"([^"]+)".+/\1/g'); \
	    dirname="$$(dirname "$$gxlname")"; \
	    cat ~/.sketch/tmp/transform.grs | sed "s/$${dirname//\//\\/}\///g" > tmp-grgen/transform.grs; \
	    cp "$$gxlname" tmp-grgen
	cp -r plugin/src/main/grgen/{AllRules.grg,ScalaAstModel.gm,rules,stages-scripts} tmp-grgen
