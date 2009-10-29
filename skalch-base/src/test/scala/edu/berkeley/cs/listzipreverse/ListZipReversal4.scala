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
        var trace = List("d","4","c","3","b","2","a","1")  // trace of angels from previous iteration

         // checking angels against a previous trace sped it up from 25s (for recur depth limited to 5) to 0s (for no limit to recursion depth)
        def ct(l:List[String]) : String = {
            synthAssertTerminal(trace != Nil) // trace better has one more entry
            val v = !!(l)
            synthAssertTerminal(v == trace.head)
            trace = trace.tail
            v
        }
        var done : Boolean  = false;
            
        for(i <- 0 to !!(5)) {  
            if (!done) {
                a = if (!!() && a!=Nil && a.tail!=Nil) a.tail else a
                b = if (!!() && b!=Nil && b.tail!=Nil) b.tail else b
                if (!!()) {
                    r = skput_and_check(ct(a)) + skput_and_check(ct(b)) :: r
                } else {
                    done = true
                }
            }
        }
        synthAssertTerminal(r == List("a1","b2","c3","d4"))    
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
