package edu.berkeley.cs.listreversejoel

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._
import java.lang.NullPointerException

/**
 * Version 2
 * ---------
 * More faithful: restricts to left-to-right traversals.
 */

class TestSketch2 extends AngelicSketch {
    val tests = Array( () )
    
    
    def main() {
    
      def reverse(l: LinkedList): LinkedList = {
        val nodeList : List[Node] = null :: l.getList
        
        var cur = l.head
        skdprint("" + cur)
        while (cur != null) {
          val t = cur.next
          cur.next = !!(nodeList)
          cur = t
        }
        l.head = !!(nodeList)
        skdprint("" + l.head)
        return l
      }
        
        try {
            var list = new LinkedList
            list.add(1)
            list.add(2)
            list.add(3)
            list.add(4)
            
            skdprint("" +  list)
            var revList = reverse(list)
            skdprint("" +  revList)
            synthAssert(revList.toList == List(4,3,2,1))
        } catch {
          case ex: java.lang.NullPointerException => synthAssert(false)
        }
    }
}

object RevList2 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new TestSketch2())
    }
}