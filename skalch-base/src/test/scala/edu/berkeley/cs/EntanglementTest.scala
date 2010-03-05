package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class EntanglementTest() extends AngelicSketch {

  val bool = List(false, true)
  val tests = Array( () )

  def main() = {
    var x : Int = 5;
    
    for(i <- 0 to !!(3)) {
      x = x - !!(10)
    }
    
    synthAssert(x == 1)
  }
}

object EntanglementTestMain {
  def main(args: Array[String]) = {
      for (arg <- args)
          Console.println(arg)
      val cmdopts = new cli.CliParser(args)
      BackendOptions.add_opts(cmdopts)
      skalch.AngelicSketchSynthesize(() => new EntanglementTest())
  }
}
