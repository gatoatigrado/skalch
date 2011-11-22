
package edu.berkeley.cs.matching

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class MatchingBSketch extends AngelicSketch {
    val tests = Array(())

    def main() {
        val groups = 6;
        val groupSize = 6;
        val n = groups * groupSize;
        val iterations = 3;

        def getFunction(): (List[Int] => Int, List[Int] => Color) = {
            var output: List[Int] = Nil
            for (value <- 0 until groups) {
                output ::= value
            }

            var valueMap: Map[List[Int], Int] = new HashMap[List[Int], Int]
            var colorMap: Map[List[Int], Color] = new HashMap[List[Int], Color]

            val function: List[Int] => Int = (input => {
                if (!valueMap.contains(input)) {
                    valueMap += input -> !!(output)
                    val c = sklast_angel_color()
                    colorMap += input -> c
                    skdprint("Creating input: " + input + " -> " + valueMap(input), c)
                }
                valueMap(input)
            })

            val colorFunction: List[Int] => Color = (input => {
                if (!colorMap.contains(input)) {
                    colorMap(input)
                } else {
                    Color.black
                }
            })

            return (function, colorFunction)
        }

        var prevGrouped = new Array[Array[Boolean]](n);
        for (i <- 0 until n) {
            prevGrouped(i) = new Array[Boolean](n);
            for (j <- 0 until n) {
                if (i == j)
                    prevGrouped(i)(j) = true;
                else
                    prevGrouped(i)(j) = false;
            }
        }

        {
            var roundGroups = new Array[Array[Int]](groups);
            for (j <- 0 until groups) {
                roundGroups(j) = new Array[Int](groupSize);
                for (k <- 0 until groupSize)
                    roundGroups(j)(k) = -1;
            }

            var groupId: Int = 0;
            for (j <- 0 until n) {
                if (j != 0 && j % groupSize == 0) {
                    groupId += 1;
                }

                val roundGroup = roundGroups(groupId);

                var added = false;
                for (k <- 0 until groupSize) {
                    if (!added) {
                        val groupMate = roundGroup(k);
                        if (groupMate != -1) {
                            synthAssert(!prevGrouped(j)(groupMate))
                            prevGrouped(j)(groupMate) = true;
                            prevGrouped(groupMate)(j) = true;
                        } else {
                            roundGroup(k) = j
                            added = true;
                        }
                    }
                }
                synthAssert(added);
            }
            skdprint(roundGroups.deep.mkString);
        }

        for (i <- 0 until iterations) {
            var roundGroups = new Array[Array[Int]](groups);
            for (j <- 0 until groups) {
                roundGroups(j) = new Array[Int](groupSize);
                for (k <- 0 until groupSize)
                    roundGroups(j)(k) = -1;
            }

            for (j <- 0 until n) {
                val groupId: Int = !!(groups);
                val roundGroup = roundGroups(groupId);

                var added = false;
                for (k <- 0 until groupSize) {
                    if (!added) {
                        val groupMate = roundGroup(k);
                        if (groupMate != -1) {
                            synthAssert(!prevGrouped(j)(groupMate))
                            prevGrouped(j)(groupMate) = true;
                            prevGrouped(groupMate)(j) = true;
                        } else {
                            roundGroup(k) = j
                            added = true;
                        }
                    }
                }
                synthAssert(added);
            }
            skdprint(roundGroups.deep.mkString);

        }

    }
}

object MatchingB {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() =>
            new MatchingBSketch())
    }
}
