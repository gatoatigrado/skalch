#!/bin/zsh

TEMP="temp_dir"

JAR="sketchrewriter.jar"
SRC="SketchRewriter.scala"
XML="scalac-plugin.xml"

TEST="PluginTest"

rm $JAR
mkdir $TEMP
scalac -d $TEMP $SRC
cp $XML $TEMP/scalac-plugin.xml
(cd $TEMP; jar cf ../$JAR .)
rm -rf $TEMP
scalac -Xplugin:$JAR $TEST.scala && scala $TEST
rm *.class
