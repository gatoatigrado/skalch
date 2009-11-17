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
class ListZipReversalSketch8 extends AngelicSketch {
    val tests = Array( () )
        
    def main() {
    try {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")
        
        var r:List[String] = Nil
        
        val up = ??(List(true, false))
                
        def descent(a:List[String], b:List[String], depth:Int=0) : (List[String],List[String]) = {
            if (!!() && depth < 4) {
                if (!up) r = skcheck(!!(List(a)).head) + skcheck(!!(List(b)).head) :: r
               
                val aaa = if (!!()) a else a.tail
                val bbb = if (!!()) b else b.tail
               
                val (aa, bb) = descent(aaa, bbb, depth+1)
               
                if (up) r = skcheck(!!(a,aa).head) + skcheck(!!(b,bb).head) :: r  
               
                return (!!(a,aa).tail,!!(b,bb).tail)  // we could make .tail optional but the only reason I could see for passing an induction variable up the resursion stack is to advance that variable, so all choices have .tail. 
            } else {
                return (a,b)   // no other values are available to the angel in the base case
            }
        }
        descent(x,y)
        synthAssert(r == List("a1","b2","c3","d4"))
    } catch { case ex : java.util.NoSuchElementException => synthAssert(false); false }
    }
}
object ListZipReverseMain8 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch8())
    }
}
