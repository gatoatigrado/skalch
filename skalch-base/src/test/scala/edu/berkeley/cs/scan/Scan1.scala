package edu.berkeley.cs.scan

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class Scan() extends AngelicSketch {
    val tests = Array( () )

    def main() : Unit = {
	try {

        val logn : Int = 3
        val n : Int = 8   // size of array
    
        def scanSpec(x : Array[Int]) : Array[Int] = {
            skdprint("in scanSpec")
            val y = new Array[Int](n)
            y(0) = x(0)
            for (val r : Int <- 1 to n-1) {
                y(r) = y(r-1)+x(r)
            } 
            return y   
        }
    
        def scanSame(x : Array[Int]) : Array[Int] = { 
            // implements scanSpec 
            skdprint("in scanSame")
            def y = new Array[Int](n);
            y(0) = x(0)
            for (val r : Int <- 1 to n-1) {
                y(r) = y(!!(n-1)) + x(!!(n-1))
            }
            return y
        }
    
        def scanSteele(x : Array[Int]) : Array[Int] = { 
            var y = new Array[Int](n)
            var x1 = x;
      
            for (val step : Int <- 0 to logn-1) {
                for (val r : Int <- 0 to n-1) {
                    if (!!(n)==0)
                        y(r) = x1(r - !!(n))+x1(r - !!(n))    
                    else {
                        y(r) = x1(r);
                    }
                }
                x1 = y.clone()
            } 
            return x1
        }
        val input : Array[Int] = Array(4,2,3,5,6,1,8,7)
        skdprint("in main")
    	  
        var r1 : Array[Int] = scanSpec(input)
        skdprint("past scanSpec")
  
        var r2 : Array[Int] = scanSame(input)
        skdprint("past scanSame")
    
        for (val i : Int <- 0 to n) { 
            synthAssert(r1(i) == r2(i)) 
        }
    } catch {
        case ex : java.util.NoSuchElementException => {synthAssert(false); true;}
    //    case ex : java.lang.ArrayIndexOutOfBoundsException => {synthAssert(false); true;}
    }
    }
}

object ScanMain1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new Scan())
    }
}

