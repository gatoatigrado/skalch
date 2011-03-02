package edu.berkeley.cs.scan

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class Scan4Sketch() extends AngelicSketch {
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
    
        def scan(x : Array[Int], spec : Array[Array[Int]], stages : Int, maxAdditions : Int,
                fanout : Int) : Array[Int] = { 
            val nodes = Array.ofDim[Tuple3[Int,Int,Int]](stages, x.length);
            var additions : Int = 0

            // set intial values
            for (val r : Int <- 0 to nodes(0).length - 1) {
                nodes(0)(r) = (r, x(r),0)
            }
            
            // angelically fill in table
            for (val r : Int <- 0 to nodes(0).length - 1) {
                for (val s : Int <- 1 to nodes.length - 1) {
                    val k : Int = !!(r+1)
                    if (k == r) {
                        nodes(s)(r) = (-1, nodes(s-1)(r)._2, 0)
                        nodes(s-1)(r) = (nodes(s-1)(r)._1, nodes(s-1)(r)._2, nodes(s-1)(r)._3 + 1)
                    } else {
                        nodes(s)(r) = (k, nodes(s-1)(r)._2 + nodes(s-1)(k)._2, 0)
                        
                        nodes(s-1)(r) = (nodes(s-1)(r)._1, nodes(s-1)(r)._2, nodes(s-1)(r)._3 + 1)
                        synthAssert(nodes(s-1)(r)._3 <= fanout)
                        
                        nodes(s-1)(k) = (nodes(s-1)(k)._1, nodes(s-1)(k)._2, nodes(s-1)(k)._3 + 1)
                        synthAssert(nodes(s-1)(k)._3 <= fanout)
                        
                        additions += 1
                        synthAssert(additions <= maxAdditions)
                    }
                    skdprint("Setting [" + s + "," + r + "] = " + nodes(s)(r))
                    if (spec(s)(r) != -1) {
                        synthAssert(nodes(s)(r)._2 == spec(s)(r))
                    }
                }
            }
            print2DArray(nodes)
            
            // check that there are no empty rows with no additions
            var seenNoAddsRow = false
            for (val s : Int <- 1 to stages - 1) {
                var noAdds = true
                for (val r : Int <- 0 to x.length - 1) {
                    if (nodes(s)(r)._1 != -1) {
                        noAdds = false
                    }
                }
                
                synthAssert(!seenNoAddsRow || noAdds)
                
                if (noAdds) {
                    seenNoAddsRow = true
                }
            }
            
            // copy over values from matrix to array
            val y = new Array[Int](x.length)
            for (val r : Int <- 0 to x.length-1) {
                y(r) = nodes(stages-1)(r)._2
            } 
            
            return y
        }
        
        def print1DArray(array : Array[Int]) {
            var outString : String = ""
            for (val i : Int <- 0 to array.length-1) {
                outString += array(i) + ","
            }
            skdprint(outString)    
        }
        
        def print2DArray(array : Array[Array[Tuple3[Int, Int, Int]]]) {
            var outString = ""
            for (val i : Int <- 0 to array.length - 1) {
                for (val j : Int <- 0 to array(i).length - 1) {
                    if (array(i)(j)._1 == -1) {
                        outString += "X "
                    } else {
                        outString  += array(i)(j)._1 + " "
                    }
                }
                outString += "\n"
            }
            skdprint(outString)            
        }
    
        //val input : Array[Int] = Array(3,5,7,11,13,17,19,23)
        val input : Array[Int] = Array(1,2,3,4,5,6,7,8)
        val mid : Array[Int] = Array(1, 3, 3, 10, 5, 11, 7, 36)
        val stages : Int = 4
        val maxAdditions : Int = 4*8
        val fanout : Int = 2
        skdprint("in main")
    	  
        var r1 : Array[Int] = scanSpec(input)
        print1DArray(r1)
        
        val r1spec = Array.ofDim[Int](stages, input.length);
        for (val r : Int <- 0 to input.length - 1) {
            for (val s : Int <- 0 to stages - 1) {
                r1spec(s)(r) = -1
            }
        }
        //r1spec(4) = mid
        r1spec(r1spec.length - 1) = r1
        
        var r2 : Array[Int] = scan(mid, r1spec, stages, maxAdditions, fanout)
        print1DArray(r2)
    }
}

object Scan4 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new Scan4Sketch())
    }
}

