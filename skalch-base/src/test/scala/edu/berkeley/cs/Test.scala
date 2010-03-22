package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._


class TestSketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
      val x1 : Int = !!(3)
      val x2 : Int = !!(3)
      val x3 : Int = !!(3)
      val x4 : Int = !!(3)
      val x5 : Int = !!(3)
  
      val x : Int = x1 + x2 + x3 + x4 + x5
    
      synthAssert(x == 5)
    }
}

object TestMain {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new TestSketch())
    }
}
