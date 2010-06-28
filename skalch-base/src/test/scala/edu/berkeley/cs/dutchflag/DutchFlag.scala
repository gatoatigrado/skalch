package edu.berkeley.cs.dutchflag

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

import sketch.entanglement._

import scala.collection.immutable.List

import java.util.ArrayList
import java.util.HashSet


class DutchFlag1Sketch extends AngelicSketch {
  val tests = Array( () )
  
  def main() {
    var input = Array('2','3','2','1','3')
    var output = Array('1','2','2','3','3')
    
    for(i <- 0 to !!(input.size * 2)) {
      swap(input, !!(input.size), !!(input.size))
      
      var debugString = ""
      for (j <- 0 to input.size - 1) {
        debugString += input(j)
      }
      skdprint(debugString)
    }
    
    for (i <- 0 to input.size - 1) {
      synthAssert(input(i) == output(i))
    }
    
    def swap(input : Array[Char], index1 : Int, index2 : Int) = {
      val temp = input(index1)
      input(index1) = input(index2)
      input(index2) = temp
    }
  }
}

object DutchFlag1 {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    val cmdopts = new cli.CliParser(args)
    BackendOptions.addOpts(cmdopts)
    skalch.AngelicSketchSynthesize(() => new DutchFlag1Sketch())
  }
}