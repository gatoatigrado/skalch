package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class ListReverse2() extends AngelicSketch {

  val bool = List(false, true)
  val tests = Array( () )

  def main() = {
    val x = List("4", "3", "2", "1")
    var r:List[String] = Nil
      
    def reverse(a:List[String]) : Unit = {
      if (!!()) {
    	r = !!(a) :: r
        synthAssert(!a.isEmpty)
        reverse(a.tail)
      }
      r
    }
    
    synthAssert(r == List("1", "2", "3", "4"))
    //synthAssert(true)
  }
}

object ListReverse {
  def main(args: Array[String]) = {
	  for (arg <- args)
		  Console.println(arg)
      val cmdopts = new cli.CliParser(args)
      BackendOptions.addOpts(cmdopts)
	  skalch.AngelicSketchSynthesize(() => new ListReverse2())
  }
}
