
package edu.berkeley.cs.matching

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

/* Try to find number of iterations where 2 people
 * dont have to meet twice */

/* Added symmetry breaking predicate */
class MatchingA3Sketch extends AngelicSketch {
    val tests = Array(())

    val groups = 4;
    val groupSize = 4;
    val n = groups * groupSize;
    val iterations = 4;

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
        
        var lastRoundGroups = new Array[Array[Int]](groups);
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
            lastRoundGroups = roundGroups;
        }
        
        var i1 = List(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3)
        var i2 = List(0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0)
        var i3 = List(0, 1, 2, 3, 2, 3, 0, 1, 3, 2, 1, 0, 1, 0, 3, 2)
        var i4 = List(0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0)
        
        for (i <- 1 to iterations) {
            var roundGroups = new Array[Array[Int]](groups);
            for (j <- 0 until groups) {
                roundGroups(j) = new Array[Int](groupSize);
                for (k <- 0 until groupSize)
                    roundGroups(j)(k) = -1;
            }
            
            var dec = List[Int]();
            for (k <- 0 until groups) {
                for (l <- 0 until groupSize) {
                    var groupId : Int = -1;
//                    if (i < 2)
//                        groupId = (k + l) % groups;
//                    else
                    groupId = !!(groups);
                    dec = dec ::: List(groupId)
                    
                    roundGroups(groupId)(k) = lastRoundGroups(k)(l);
                    
//                    skdprint("" + groupId);
//                    skdprint(roundGroups.deep.mkString);
                    
                    
                    for (m <- 0 until k) {
                        val p1 = roundGroups(groupId)(k);
                        val p2 = roundGroups(groupId)(m);
                        
                        synthAssert(prevGrouped(p1)(p2) == -1)
                    }
                }
                
                for (l <- 0 until groups) {
                    synthAssert(roundGroups(l)(k) != -1)
                }
                
            }
            
            if (i == 1) {
                synthAssert(dec == i1)
            } else if (i == 2) {
                synthAssert(dec == i2)
            }

            skdprint(roundGroups.deep.mkString);

            
            for (m <- 0 until groups) {
                for (n <- 0 until groupSize) {
                    for (o <- n + 1 until groupSize) {
                        val p1 = roundGroups(m)(n);
                        val p2 = roundGroups(m)(o);
                        
                        synthAssert(prevGrouped(p1)(p2) == -1)
                        prevGrouped(p1)(p2) = i;
                        prevGrouped(p2)(p1) = i;
                    }
                }
            }
            
            

//            for (j <- 0 until n) {
//                val groupId: Int = !!(groups);
////                if (j == 0)
////                    synthAssert(groupId == 0);
//
//                val roundGroup = roundGroups(groupId);
//
//                var added = false;
//                for (k <- 0 until groupSize) {
//                    if (!added) {
//                        val groupMate = roundGroup(k);
//                        if (groupMate != -1) {
//                            synthAssert(prevGrouped(j)(groupMate) == -1)
//                            prevGrouped(j)(groupMate) = i;
//                            prevGrouped(groupMate)(j) = i;
//                        } else {
//                            roundGroup(k) = j
//                            added = true;
//                        }
//                    }
//                }
//                synthAssert(added);
//            }

                        
//            for (j <- 0 until groups) {
//                for (k <- 0 until groupSize) {
//                    sktrace(roundGroups(j)(k), n);
//                }
//            }

//            for (j <- 0 until groups) {
//                for (k <- j + 1 until groups) {
//                    synthAssert(roundGroups(j)(0) < roundGroups(k)(0));
//                }
//            }

            skdprint(roundGroups.deep.mkString);
            printRoundGroupMatrix(roundGroups);
            lastRoundGroups = roundGroups;
        }
        
        printMatrix(prevGrouped);

    }
    
    def traceMatrix(matrix : Array[Array[Int]]) {
        for (i <- 0 until n) {
            for (j <- 0 until n) {
                sktrace(matrix(i)(j), iterations);
            }
        }
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

object MatchingA3 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() =>
            new MatchingA3Sketch())
    }
}
