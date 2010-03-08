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
class ListZipReversalSketch10 extends AngelicSketch {
    val tests = Array( () )
        
    def main() {
    try {
        val x = List("a", "b", "c", "d")
        val y = List("4", "3", "2", "1")
        
        var r:List[String] = Nil
        
        def descent(a:List[String], b:List[String], depth:Int=0) : List[String] = {
            if (!!() && depth < 4) {
                val bb = descent(a.tail, b, depth+1)
                r = skcheck(a.head) + skcheck(bb.head) :: r  
                return bb.tail
            } else {
                return b
            }
        }
        descent(x,y)
        synthAssert(r == List("a1","b2","c3","d4"))
    } catch { case ex : java.util.NoSuchElementException => synthAssert(false); false }
    }
}
object ListZipReverseMain10 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new ListZipReversalSketch10())
    }
}
