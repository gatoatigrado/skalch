package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class EntanglementTest() extends AngelicSketch {

  val bool = List(false, true)
  val tests = Array( () )

  def main() = {
    var x : Int = 0;
    
    for(i <- 0 to 4) {
      x = x + !!(3)
    }
    
    synthAssert(x == 5)
  }
}

object EntanglementTestMain {
  def main(args: Array[String]) = {
      for (arg <- args)
          Console.println(arg)
      val cmdopts = new cli.CliParser(args)
      BackendOptions.addOpts(cmdopts)
      skalch.AngelicSketchSynthesize(() => new EntanglementTest())
  }
}
