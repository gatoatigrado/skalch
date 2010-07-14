package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class SEntanglementTest() extends AngelicSketch {

  val bool = List(false, true)
  val tests = Array( () )

  def main() = {
      
      
    val a0 : Int = !!(0,1)
    val a1 : Int = !!(0,1)
    val a2 : Int = !!(0,1)
    val a3 : Int = !!(0,1)
    val a4 : Int = !!(0,1)
    
    if (a1 == 1) {
        synthAssert (a2 + a3 <= 1)
    } else {
        synthAssert (a2 + a3 == 2)
    }
    synthAssert (a4 == 1 - a1)
    synthAssert(a0 + a1 + a2 > 0)
  }
}

object SmallTestMain {
  def main(args: Array[String]) = {
      for (arg <- args)
          Console.println(arg)
      val cmdopts = new cli.CliParser(args)
//      BackendOptions.addOpts(cmdopts)
      skalch.AngelicSketchSynthesize(() => new SEntanglementTest())
  }
}
