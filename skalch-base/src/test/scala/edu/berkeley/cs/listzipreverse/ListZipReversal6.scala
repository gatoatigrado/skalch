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

class ListZipReversalSketch6 extends AngelicSketch {
    val tests = Array( () )
    
    def sublists (a:List[String]):List[List[String]] = if (a==Nil) List(Nil) else a :: sublists(a.tail)
    
    def main() {
    try {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")
        
        var r:List[String] = Nil
        
        val up = !!() // are we doing "work" on the way up from the recursion or down?
    
        def descent(a:List[String], b:List[String], depth:Int=0) : (List[String],List[String]) = {
            if (!!() && depth < 4) {
                if (!up) r = skcheck(!!(List(a)).head) + skcheck(!!(List(b)).head) :: r
    
                val aaa = if (!!()) a else a.tail
                // these values don't matter (becasue the retunr value is under angelic control)
                val bbb = if (true/*!!()*/) b else b.tail
                val (aa, bb) = descent(aaa, bbb, depth+1)
    
                // always select bb is correct
                // always select b is not correct (ie b is sometimes the correct choice)
                if (up) r = skcheck(!!(List(a,aa)).head) + skcheck(bb/*!!(List(b,bb))*/.head) :: r  
               
                // the first two values are always 1,2
                // the last value of !!(sublists(y)) returned to main does not matter
                return (!!(sublists(x)),bb.tail/*!!(sublists(y))*/)
            } else {
                // value is always 0
                return (!!(sublists(x)),b/*!!(sublists(y))*/)
            }
        }
        descent(x,y)
        synthAssert(r == List("a1","b2","c3","d4"))
    
    } catch { case ex : java.util.NoSuchElementException => synthAssert(false); false }
    }
}

object ListZipReverseMain6 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch6())
    }
}
