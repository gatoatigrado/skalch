
package edu.berkeley.cs.matching

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

/* Try to find number of iterations where 2 people
 * dont have to meet twice */

/* Added symmetry breaking predicate */
class MatchingMatrixSketch extends AngelicSketch {
    val tests = Array(())

    val groups = 3;
    val groupSize = 3;
    val n = groups * groupSize;
    val iterations = 3;
    
    def main() {
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
            printRoundGroupMatrix(roundGroups);
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
                if (j == 0)
                    synthAssert(groupId == 0);

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
            printRoundGroupMatrix(roundGroups);
        }
        
        printMatrix(prevGrouped);

    }
    
    def printMatrix(matrix : Array[Array[Int]]) {
        var str = "";
        for (i <- 0 until n) {
            for (j <- 0 until n) {
                str += matrix(i)(j);
            }
            str += "\n";
        }
        skdprint(str);    
    }

    def printRoundGroupMatrix(roundGroups : Array[Array[Int]]) {
        var roundGroupMatrix = new Array[Array[Int]](n);
        for (i <- 0 until n) {
            roundGroupMatrix(i) = new Array[Int](n);
            for (j <- 0 until n)
                roundGroupMatrix(i)(j) = 0;
        }

        for (i <- 0 until groups) {
            for (j <- 0 until groupSize) {
                for (k <- j until groupSize) {
                    val p1 = roundGroups(i)(j);
                    val p2 = roundGroups(i)(k);
                    roundGroupMatrix(p1)(p2) = i + 1;
                    roundGroupMatrix(p2)(p1) = i + 1;
                }
            }
        }
        printMatrix(roundGroupMatrix);
    }
}

object MatchingMatrix {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() =>
            new MatchingA1Sketch())
    }
}
