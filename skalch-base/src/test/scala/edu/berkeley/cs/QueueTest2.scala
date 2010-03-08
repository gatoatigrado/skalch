package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class QueueTest2() extends AngelicSketch {

  val tests = Array( () )

  def main() = {
    var x : Int = !!(100)
    var y : Int = 0
    while (x > 0) {
    	skput("" + x)
    	skcheck("" + x)
    	x = x - 1
    	y = y + 1
    }
    synthAssert(y < 100)
  }
}

object QueueTest2Main {
  def main(args: Array[String]) = {
	  for (arg <- args)
		  Console.println(arg)
      val cmdopts = new cli.CliParser(args)
      BackendOptions.addOpts(cmdopts)
	  skalch.AngelicSketchSynthesize(() => new QueueTest2())
  }
}
