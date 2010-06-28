package edu.berkeley.cs

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

import sketch.entanglement._

import scala.collection.immutable.List

import java.util.ArrayList
import java.util.HashSet


class TestSketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {
      val numTraces : Int = !!(100)
      val numAngels : Int = 5
      val numValues : Int = 3
      
      var angels : List[DynAngel] = List[DynAngel]()
      for (i <- 0 to numAngels) {
        angels = new DynAngel(i, 0) :: angels
      }
      
      var traces : ArrayList[Trace] = new ArrayList[Trace]()
      for (i <- 0 to numTraces) {
        var trace : Trace = new Trace()
        for (j <- 0 to numAngels) {
          trace.addEvent(j, 0, numValues, !!(numValues))
        }
        traces.add(trace)
      }
      
      var ea : EntanglementAnalysis = new EntanglementAnalysis(traces)
      
      synthAssert(angels.size > 2)
      val a : DynAngel = angels(0)
      val b : DynAngel = angels(1)
      val c : DynAngel = angels(2)
      
      var A : HashSet[DynAngel] = new HashSet[DynAngel]()
      var B : HashSet[DynAngel] = new HashSet[DynAngel]()
      
      A.add(a)
      B.add(b)
      B.add(c)
      
      for (i <- 3 to numAngels) {
        if (!!()) {
          A.add(angels(i))
        } else {
          B.add(angels(i))
        }
      }
      
      val result : EntanglementComparison = ea.compareTwoSubtraces(A, B, true)
      synthAssert(!result.isEntangled)
      
      var Q : HashSet[DynAngel] = new HashSet[DynAngel]()
      var R : HashSet[DynAngel] = new HashSet[DynAngel]()
      
      Q.add(a)
      Q.add(c)
      R.add(b)
      
      for (i <- 3 to numAngels) {
        if (!!()) {
          if (!!()) {
            Q.add(angels(i))
          } else {
            R.add(angels(i))
          }
        }
      }    

      val result1 : EntanglementComparison = ea.compareTwoSubtraces(Q, R, true)
      synthAssert(result1.isEntangled)
      
      Q.remove(a)
      val result2 : EntanglementComparison = ea.compareTwoSubtraces(Q, R, true)
      synthAssert(!result2.isEntangled)
      
      Q.add(a)
      Q.remove(b)
      val result3 : EntanglementComparison = ea.compareTwoSubtraces(Q, R, true)
      synthAssert(!result3.isEntangled)
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