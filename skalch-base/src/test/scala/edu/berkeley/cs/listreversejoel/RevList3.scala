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

class TestSketch3 extends AngelicSketch {
    val tests = Array( () )
    
    
    def main() {
    
        def reverse(l: LinkedList): LinkedList = {
            var prev: Node = null
            var cur = l.head
            while (cur != null) {
              val t = cur.next
              cur.next = prev
              prev = cur
              cur = t
            }
            l.head = prev
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

object RevList3 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new TestSketch3())
    }
}