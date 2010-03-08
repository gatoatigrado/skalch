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
    
    def sublists (a:List[String]):List[List[String]] = if (a==Nil) List(Nil) else a :: sublists(a.tail)
    
    def main() {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")
        
        var r:List[String] = Nil
        val up = !!() // do we need to do "work" (ie consing) on the way up from the recursion or down?

        def descent(a:List[String], b:List[String], depth:Int=0) : (List[String],List[String]) = {
            if (!!() && depth < 4) {
                if (!up) r = skput_check(!!(x)) + skput_check(!!(y)) :: r

                val aaa = if (!!()) a else a.tail
                val bbb = if (!!()) b else b.tail
                val (aa, bb) = descent(aaa, bbb, depth+1)

                if (up) r = skput_check(!!(x)) + skput_check(!!(y)) :: r

                return (!!(sublists(x)),!!(sublists(y)))
            } else {
                return (!!(sublists(x)),!!(sublists(y)))
            }
        }

        descent(x,y)
        synthAssert(r == List("a1","b2","c3","d4"))
    }
}

object ListZipReverseMain5 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch5())
    }
}
