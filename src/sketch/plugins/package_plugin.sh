#!/bin/zsh

TEMP="working_dir"
JAR="sketchrewriter.jar"
SRC="SketchRewriter.scala"
TEST="PluginTest"

rm $JAR
mkdir $TEMP
fsc -d $TEMP $SRC
cp scalac-plugin.xml $TEMP
(cd $TEMP; jar cf ../$JAR .)
rm -rf $TEMP
scalac -Xplugin:$JAR $TEST.scala && scala $TEST
