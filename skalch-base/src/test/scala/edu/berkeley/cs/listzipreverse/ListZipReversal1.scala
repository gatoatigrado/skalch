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

class ListZipReversalSketch1 extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")
        var r:List[String] = Nil

        for(i <- 0 to !!(6)) {  // once backtracking is fixed, this should be while(!!())
            r = skput_and_check(!!(x)) + skput_and_check(!!(y)) :: r  // (!!,!!) trace ="d","1",...
        }
        synthAssertTerminal(r == List("a1","b2","c3","d4"))
    }
}

object ListZipReverseMain1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch1())
        }
    }
