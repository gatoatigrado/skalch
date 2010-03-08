package edu.berkeley.cs.listreversejoel

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._
import java.lang.NullPointerException

/**
 * Version 1 
 * ---------
 * Simple, very unfaithful.
 */

class TestSketch extends AngelicSketch {
    val tests = Array( () )
    
    
    def main() {
    
        def reverse(l: LinkedList) : LinkedList = {
            val len = l.length
            val nodeList = l.getList
            for (i <- 0 to !!(len)) {
                !!(nodeList).next = !!(null :: nodeList)
            }
            var n : Node = !!(nodeList)
            l.head = n
            skdprint("" + l.head + ", " + n)
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

object RevList1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new TestSketch())
    }
}


