package edu.berkeley.cs.scan

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class Scan7Sketch() extends AngelicSketch {
    val tests = Array( () )

    def main() : Unit = {
        
        class BitVector(val length : Int) {
            val vector : Array[Boolean] = new Array[Boolean](length)
            
            def apply(i : Int) : Boolean = {
                return vector(i)
            }
            
            def update(i : Int, x : Boolean) {
                vector(i) = x
            }
            
            def +(other : BitVector) : BitVector = {
                val sum = new BitVector(length)
                var seenBitOther = false
                for (i <- 0 until length) {
                    synthAssert(!this(i) || !other(i) )
                    sum(i) = this(i) || other(i)
                    
                    if (!seenBitOther && other(i)) {
                        seenBitOther = true
                        synthAssert(this(i-1))
                    }
                }
                return sum
            }
            
            def ==(other : BitVector) : Boolean = {
                if (other.length != length) {
                    return false
                }
                for (i <- 0 until length) {
                    if (other(i) != this(i)) {
                        return false
                    }
                }
                return true
            }
            
            override def toString : String = {
                var sum = 0;
                var curNum = 1;
                for (i <- 0 until length) {
                    if (vector(i)) {
                        sum += curNum
                    }
                    curNum *= 2
                }
                return "" + sum
            }
        }
     
        def scanSpec(x : Array[BitVector]) : Array[BitVector] = {
            val y = new Array[BitVector](x.length)
            y(0) = x(0)
            for (val r : Int <- 1 until x.length) {
                y(r) = y(r-1) + x(r)
            } 
            return y
        }
    
        def scan(x : Array[BitVector], spec : Array[BitVector], stages : Int,
                maxAdditions : Int, fanout : Int) : Array[BitVector] = { 
            val nodes = Array.ofDim[Tuple3[Int,BitVector,Int]](stages, x.length);
            var additions : Int = 0

            // set intial values
            for (val r : Int <- 0 until nodes(0).length) {
                nodes(0)(r) = (r, x(r),0)
            }
            
            // angelically fill in table
            for (val r : Int <- 0 until nodes(0).length) {
                for (val s : Int <- 1 until nodes.length) {
                    val k : Int = !!(r+1)
                    if (k == r) {
                        nodes(s)(r) = (-1, nodes(s-1)(r)._2, 0)
                        nodes(s-1)(r) = (nodes(s-1)(r)._1, nodes(s-1)(r)._2, nodes(s-1)(r)._3 + 1)
                    } else {
                        nodes(s)(r) = (k, nodes(s-1)(k)._2 + nodes(s-1)(r)._2, 0)
                        
                        nodes(s-1)(r) = (nodes(s-1)(r)._1, nodes(s-1)(r)._2, nodes(s-1)(r)._3 + 1)
                        synthAssert(nodes(s-1)(r)._3 <= fanout)
                        
                        nodes(s-1)(k) = (nodes(s-1)(k)._1, nodes(s-1)(k)._2, nodes(s-1)(k)._3 + 1)
                        synthAssert(nodes(s-1)(k)._3 <= fanout)
                        
                        additions += 1
                        synthAssert(additions <= maxAdditions)
                    }
                    skdprint("Setting [" + s + "," + r + "] = " + nodes(s)(r))
                }
                synthAssert(nodes(nodes.length-1)(r)._2 == spec(r))               
            }
            print2DArray(nodes)
            
            // check that there are no empty rows with no additions
//            var seenNoAddsRow = false
//            for (val s : Int <- 1 until stages) {
//                var noAdds = true
//                for (val r : Int <- 0 until x.length) {
//                    if (nodes(s)(r)._1 != -1) {
//                        noAdds = false
//                    }
//                }
//                
//                synthAssert(!seenNoAddsRow || noAdds)
//                
//                if (noAdds) {
//                    seenNoAddsRow = true
//                }
//            }
            
            // copy over values from matrix to array
            val y = new Array[BitVector](x.length)
            for (val r : Int <- 0 until x.length) {
                y(r) = nodes(stages-1)(r)._2
            } 
            
            return y
        }
        
        def print1DArray(array : Array[BitVector]) {
            var outString : String = ""
            for (val i : Int <- 0 until array.length) {
                outString += array(i) + ","
            }
            skdprint(outString)    
        }
        
        def print2DArray(array : Array[Array[Tuple3[Int, BitVector, Int]]]) {
            var outString = ""
            for (val i : Int <- 0 until array.length) {
                for (val j : Int <- 0 until array(i).length) {
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
        
        def getBitVector(num : Int, length : Int) : BitVector = {
            val vector = new BitVector(length)
            var curNum = num
            var i = 0
            
            while (curNum > 0) {
                if (curNum % 2 == 1) {
                    vector(i) = true
                }
                curNum /= 2
                i += 1
            }
            
            return vector
        }
    
        val input : Array[Int] = Array(1,2,4,8,16,32,64,128
                ,256,512,1024,2048,4096,8192,16384,32768)
        val vectorInput : Array[BitVector] = new Array[BitVector](input.length)
        for (i <- 0 until vectorInput.length) {
            //vectorInput(i) = getBitVector(input(i), input.length)
            vectorInput(i) = new BitVector(input.length)
            vectorInput(i)(i) = true
        }
        
        val mid : Array[Int] = Array(1,2,6,14,30,32,96,224
                ,496,512,1536,3584,8176,8192,24576,57344)
        val vectorMid : Array[BitVector] = new Array[BitVector](mid.length)
        for (i <- 0 until vectorMid.length) {
            vectorMid(i) = getBitVector(mid(i), mid.length)
        }
        
        val stages : Int = 4
        val maxAdditions : Int = 16*4
        val fanout : Int = 3
        skdprint("in main")
    	  
        var spec : Array[BitVector] = scanSpec(vectorInput)
        print1DArray(spec)
        
        
        var r2 : Array[BitVector] = scan(vectorMid, spec, stages, maxAdditions, fanout)
        print1DArray(r2)
    }
}

object Scan7 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new Scan7Sketch())
    }
}

