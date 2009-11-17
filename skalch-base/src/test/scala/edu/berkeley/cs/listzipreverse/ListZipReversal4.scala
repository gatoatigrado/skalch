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

class ListZipReversalSketch4 extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")

        var a = x
        var b = y
        
        var r:List[String] = Nil
        val up = !!() // do we need to do "work" (ie consing) on the way up from the recursion or down?

        def descent(depth: Int) : Unit = {
            if (!!() && depth < 6) {
                if (!up) r = skput_and_check(!!(x)) + skput_and_check(!!(y)) :: r
                descent(depth + 1)
                if (up)  r = skput_and_check(!!(x)) + skput_and_check(!!(y)) :: r
            }
        }

        descent(0)
        synthAssert(r == List("a1","b2","c3","d4"))
    }
}

object ListZipReverseMain4 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch4())
    }
}
