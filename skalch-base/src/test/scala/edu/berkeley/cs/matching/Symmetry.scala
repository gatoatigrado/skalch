
package edu.berkeley.cs.matching

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class SymmetrySketch extends AngelicSketch {
    val tests = Array(())

    def main() {

        val n = 4;

        var permutation : Array[Int] = new Array[Int](n);
        for (i <- 0 until n) {
            permutation(i) = -1;
        }
            
        for (i <- 0 until n) {
            val index : Int = !!(n);
            synthAssert(permutation(index) == -1)
            permutation(index) = i;
        }
        
        for (i <- 0 until n) {
            sktrace(permutation(i), n);
        }
        
        skdprint(permutation.deep.mkString);
    }
}

object Symmetry {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() =>
            new SymmetrySketch())
    }
}
