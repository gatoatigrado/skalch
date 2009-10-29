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

class ListZipReversalSketch5 extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")

        var a = x
        var b = y
        
        var r:List[String] = Nil
        val up = !!() // do we need to do "work" (ie consing) on the way up from the recursion or down?

        def descent() : Unit = {
            if (!!()) {
                if (!up) r = skput_and_check(!!(x)) + skput_and_check(!!(y)) :: r   // angels that appear in the previous version (and have not been refined) are now replaced with ct(), which checks that the value generated is the same as in the safe trace from previous version
                descent()
                if (up)  r = skput_and_check(!!(x)) + skput_and_check(!!(y)) :: r
            }
        }

        descent()
        synthAssertTerminal(r == List("a1","b2","c3","d4"))   
    }
}

object ListZipReverseMain5 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch5())
    }
}
