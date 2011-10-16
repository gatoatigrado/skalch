
package edu.berkeley.cs.matching

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

/* Try to find number of iterations where 2 people
 * dont have to meet twice */

/* Added symmetry breaking predicate */
class Matching1Sketch extends AngelicSketch {
    val tests = Array(())

    def main() {
        val groups = 3;
        val groupSize = 3;
        val n = groups * groupSize;
        val iterations = 3;

        var prevGrouped = new Array[Array[Int]](n);
        for (i <- 0 until n) {
            prevGrouped(i) = new Array[Int](n);
            for (j <- 0 until n) {
                if (i == j)
                    prevGrouped(i)(j) = 0;
                else
                    prevGrouped(i)(j) = -1;
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
                            synthAssert(prevGrouped(j)(groupMate) == -1)
                            prevGrouped(j)(groupMate) = 0;
                            prevGrouped(groupMate)(j) = 0;
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

        for (i <- 1 to iterations) {
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
                            synthAssert(prevGrouped(j)(groupMate) == -1)
                            prevGrouped(j)(groupMate) = i;
                            prevGrouped(groupMate)(j) = i;
                        } else {
                            roundGroup(k) = j
                            added = true;
                        }
                    }
                }
                synthAssert(added);
            }

            for (j <- 0 until groups) {
                for (k <- j + 1 until groups) {
                    synthAssert(roundGroups(j)(0) < roundGroups(k)(0));
                }
            }
            skdprint(roundGroups.deep.mkString);
        }
        
        var matrix = "";
        for (i <- 0 until n) {
            for (j <- 0 until n) {
                matrix += prevGrouped(i)(j);
            }
            matrix += "\n";
        }
        skdprint(matrix);
    }
}

object Matching1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() =>
            new Matching1Sketch())
    }
}
