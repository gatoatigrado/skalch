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

class ListZipReversalSketch2 extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")

        var a = x
        var b = y
        
        var r:List[String] = Nil

        for(i <- 0 to !!(5)) {  
            a = if (!!() && a!=Nil && a.tail!=Nil) a.tail else a
            b = if (!!() && b!=Nil && b.tail!=Nil) b.tail else b
            if (!!()) r = skput_check(!!(a)) + skput_check(!!(b)) :: r 
        }
        synthAssert(r == List("a1","b2","c3","d4"))    
    }
}

object ListZipReverseMain2 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch2())
        }
    }
