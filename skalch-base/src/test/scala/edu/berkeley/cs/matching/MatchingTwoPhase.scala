
package edu.berkeley.cs.matching

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class MatchingTwoPhaseSketch extends AngelicSketch {
    val tests = Array(())

    def main() {
        val groups = 4;
        val groupSize = 3;
        val n = groups * groupSize;
        
        val iterationNoDup = 3;
        val iterationFinish = 3;

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

        for (i <- 0 until iterationNoDup) {
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

//            for (j <- 0 until groups) {
//                for (k <- j + 1 until groups) {
//                    synthAssert(roundGroups(j)(0) < roundGroups(k)(0));
//                }
//            }
            skdprint(roundGroups.deep.mkString);

        }

        var oldNumMatched = groupSize * groupSize * groups;
        for (i <- 0 until iterationFinish) {
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

            var numMatched = 0;
            for (i <- 0 until n) {
                for (j <- 0 until n) {
                    numMatched += 1;
                }
            }
//            synthAssert(numMatched > ((1 + i) * n * n) / (1 + iterations))
            synthAssert(numMatched > oldNumMatched + 1 || numMatched == n * n);
            oldNumMatched = numMatched;
        }
        for (i <- 0 until n) {
            for (j <- 0 until n) {
                synthAssert(prevGrouped(i)(j));
            }
        }
    }
}

object MatchingTwoPhase {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() =>
            new MatchingTwoPhaseSketch())
    }
}
