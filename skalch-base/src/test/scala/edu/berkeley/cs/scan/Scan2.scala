package edu.berkeley.cs.scan

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class Scan2Sketch() extends AngelicSketch {
    val tests = Array( () )

    def main() : Unit = {
    
        def scanSpec(x : Array[Int]) : Array[Int] = {
            val y = new Array[Int](x.length)
            y(0) = x(0)
            for (val r : Int <- 1 to x.length-1) {
                y(r) = y(r-1)+x(r)
            } 
            return y   
        }
    
        def scan(x : Array[Int], spec : Array[Int], stages : Int) : Array[Int] = { 
            val nodes = Array.ofDim[Tuple2[Int,Int]](x.length, stages);
            for (val r : Int <- 0 to x.length - 1) {
                nodes(r)(0) = (r, x(r))
            }
            for (val r : Int <- 0 to x.length - 1) {
                for (val s : Int <- 1 to stages - 1) {
                    val k : Int = !!(r+1)
                    if (k == r) {
                        nodes(r)(s) = (k, nodes(r)(s-1)._2)
                    } else {
                        nodes(r)(s) = (k, nodes(r)(s-1)._2 + nodes(k)(s-1)._2)
                    }
                    skdprint("Setting [" + r + "," + s + "] = " + nodes(r)(s))
                }
                synthAssert(nodes(r)(stages-1)._2 == spec(r))
            }
            
            val y = new Array[Int](x.length)
            for (val r : Int <- 0 to x.length-1) {
                y(r) = nodes(r)(stages-1)._2
            } 
            
            return y
        }
    
        val input : Array[Int] = Array(4,2,3,5,6,1,8,7)
        skdprint("in main")
    	  
        var r1 : Array[Int] = scanSpec(input)
        var r1String : String = ""
        for (val r : Int <- 0 to r1.length-1) {
            r1String += r1(r) + ","
        }
        skdprint("spec: " + r1String)
    
        var r2 : Array[Int] = scan(input, r1, 5)
        var r2String : String = ""
        for (val r : Int <- 0 to r2.length-1) {
            r2String += r2(r) + ","
        }
        skdprint("scan: " + r2String)
    
        for (val i : Int <- 0 to r1.length-1) { 
            synthAssert(r1(i) == r2(i)) 
        }
    }
}

object Scan2 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new Scan2Sketch())
    }
}

