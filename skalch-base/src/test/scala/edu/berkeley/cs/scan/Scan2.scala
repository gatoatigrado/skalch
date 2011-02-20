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
    
        def scan(x : Array[Int], spec : Array[Int], _stages : Int, maxAdditions : Int,
                fanout : Int) : Array[Int] = { 
            val stages = _stages + 1
            val nodes = Array.ofDim[Tuple3[Int,Int,Int]](x.length, stages);
            var additions : Int = 0
            for (val r : Int <- 0 to x.length - 1) {
                nodes(r)(0) = (r, x(r),0)
            }
            for (val r : Int <- 0 to x.length - 1) {
                for (val s : Int <- 1 to stages - 1) {
                    val k : Int = !!(r+1)
                    if (k == r) {
                        nodes(r)(s) = (-1, nodes(r)(s-1)._2, 0)
                        nodes(r)(s-1) = (nodes(r)(s-1)._1, nodes(r)(s-1)._2, nodes(r)(s-1)._3 + 1)
                    } else {
                        nodes(r)(s) = (k, nodes(r)(s-1)._2 + nodes(k)(s-1)._2, 0)
                        
                        nodes(r)(s-1) = (nodes(r)(s-1)._1, nodes(r)(s-1)._2, nodes(r)(s-1)._3 + 1)
                        synthAssert(nodes(r)(s-1)._3 <= fanout)
                        
                        nodes(k)(s-1) = (nodes(k)(s-1)._1, nodes(k)(s-1)._2, nodes(k)(s-1)._3 + 1)
                        synthAssert(nodes(k)(s-1)._3 <= fanout)
                        
                        additions += 1
                        synthAssert(additions <= maxAdditions)
                    }
                    skdprint("Setting [" + r + "," + s + "] = " + nodes(r)(s))
                }
                synthAssert(nodes(r)(stages-1)._2 == spec(r))
            }
            
            var seenNoAddsRow = false
            for (val s : Int <- 1 to stages - 1) {
                var noAdds = true
                for (val r : Int <- 0 to x.length - 1) {
                    if (nodes(r)(s)._1 != -1) {
                        noAdds = false
                    }
                }
                
                synthAssert(!seenNoAddsRow || noAdds)
                
                if (noAdds) {
                    seenNoAddsRow = true
                }
            }
            
            var outString = ""
            for (val s : Int <- 1 to stages - 1) {
                for (val r : Int <- 0 to x.length - 1) {
                    if (nodes(r)(s)._1 == -1) {
                        outString += "X "
                    } else {
                        outString  += nodes(r)(s)._1 + " "
                    }
                }
                outString += "\n"
            }
            skdprint(outString)
            
            val y = new Array[Int](x.length)
            for (val r : Int <- 0 to x.length-1) {
                y(r) = nodes(r)(stages-1)._2
            } 
            
            return y
        }
    
        val input : Array[Int] = Array(3,5,7,11,13,17,19,23)
        val stages : Int = 8
        val maxAdditions : Int = 8*8
        val fanout : Int = 2
        skdprint("in main")
    	  
        var r1 : Array[Int] = scanSpec(input)
        var r1String : String = ""
        for (val r : Int <- 0 to r1.length-1) {
            r1String += r1(r) + ","
        }
        skdprint("spec: " + r1String)
    
        var r2 : Array[Int] = scan(input, r1, stages, maxAdditions, fanout)
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

