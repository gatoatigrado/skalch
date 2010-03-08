package edu.berkeley.cs.listzipreverse

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

/*
 * 
 * Constraints:
 * Angelic choices:
 *   
 * Lessens learned:
 */

class ListZipReversalSketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")
        var r:List[String] = Nil
        
        r = !!(x) + !!(y) :: r  // (!!,!!) = "d","4"
        r = !!(x) + !!(y) :: r  // (!!,!!) = "c","3"
        r = !!(x) + !!(y) :: r  // (!!,!!) = ...
        r = !!(x) + !!(y) :: r  // (!!,!!) = ...
        synthAssert(r == List("a1","b2","c3","d4"))
    }
}

object ListZipReverseMain {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch())
        }
    }
