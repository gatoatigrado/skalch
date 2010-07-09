
package edu.berkeley.cs.bipartite

import skalch.AngelicSketch
import sketch.util._

class BipartiteSketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {

    	class Vertex(name : String) {
	      val MAX_OUT = 5

	      var id : String = name
	      var edges : Array[Vertex] = new Array[Vertex](MAX_OUT)
	      var numedges : Int = 0
	      var inset : Int = -1  // uninit

	      override def toString() : String = {
	           var returnString : String = id
		   returnString += "["
		   for (j <- 0 to numedges - 1) returnString += edges(j).id 
		   returnString += "]"
	           return returnString
	      }

	      def addEdge(target : Vertex) {
	      	  edges(numedges) = target
	      	  numedges += 1
          }
    	}

	class Edge(s : Vertex, t : Vertex) {
	     var src : Vertex = s
             var dest : Vertex = t
	     var handled : Boolean = false

             override def toString() : String = {
                   var returnString : String = "<"
                   returnString += src
                   returnString += ","
                   returnString += dest
                   returnString += ">"
		   return returnString
             }
        }

	//	Idea: we will keep marks. we will assume up to two marks can be updated per go. we will force
	//	the angels to reveal to us the order in which edges are picked

        class Graph {
          val MAXV = 10
          val MAXE = 50
	      var nv : Int = 0 // number of vertices
	      var ne : Int = 0 // number of edges

          var vertices : Array[Vertex] = new Array[Vertex](MAXV)
          var edges : Array[Edge] = new Array[Edge](MAXE)

          
          def checkBipartiteV0() = {
               for (j <- 0 to ne - 1) { // repeat #edges times
                     val e : Edge = edges(!!(ne))
                     synthAssert (! e.handled)
                     skdprint("Selecting Edge " + e)
                     e.handled = true
                     e.src.inset = !!(0,1)
                     e.dest.inset = 1 - e.src.inset // !!(0,1) 
                     skdprint("Polarity: (" + e.src.inset + "," + e.dest.inset + ")")                                       
                     synthAssert (beta())
               }
          }
          
          // wavefront
          def checkBipartiteV1(root : Vertex) = {
               for (j <- 0 to ne - 1) { // repeat #edges times
                     val e : Edge = edges(!!(ne))
                     synthAssert (! e.handled)
                     e.handled = true
                     if (j == 0) { 
                         synthAssert (e.src == root)
                         e.src.inset = 0
                     } else {
                         synthAssert ( e.src.inset != -1)  
                         // src of this edge is a vertex whose polarity has been fixed already
                     }
                     skdprint("Selecting Edge " + e)
                     e.dest.inset = 1 - e.src.inset                                   
                     synthAssert (beta())
               }
          }
                
          // wavefront, polarity angels, flip computation
          def checkBipartiteV2(root : Vertex) = {
               var curp : Int = 1
               var flipcnt : Int = 0;
               for (j <- 0 to ne - 1) { // repeat #edges times
                     val e : Edge = edges(!!(ne))
                     synthAssert (! e.handled)
                     e.handled = true
                     if (j == 0) { 
                         synthAssert (e.src == root)
                         e.src.inset = 0
                     } else {
                         synthAssert ( e.src.inset != -1)  
                         // src of this edge is a vertex whose polarity has been fixed already
                     }
                     skdprint("Selecting Edge " + e)
                     e.dest.inset = !!(0,1) // 1 - e.src.inset
                     if (e.dest.inset != curp) {
                         curp = e.dest.inset
                         flipcnt = flipcnt + 1
                         // synthAssert (flipcnt < 4) // don't flip too often
                     }
                     synthAssert (beta())
               }
               skdprint("flips " + flipcnt)
          }
          
          // at this time i am seeing a lot of 
          
	      def beta() : Boolean = {
	      	  for (j <- 0 to ne - 1) {
	      	      val e : Edge = edges(j)
	      	      if (e.handled) {
	      	       if (e.src.inset == 1)
	      	          if (e.dest.inset != 0)
	      	               return false
	      	       if (e.src.inset == 0)
	      	          if (e.dest.inset != 1)
	      	               return false
	      	      }
	      	  } 
		      return true
	      }
	      
	      def printBipartite() = {
	           skdprint("Set is:")
	           for (j <- 0 to nv - 1) {
	                 if (vertices(j).inset == 1)
	                     skdprint(vertices(j).toString)
	           }
	      }
	       
	       
          def addVertex(v : Vertex) { 
                vertices(nv) = v
                nv = nv + 1
              }

              def addEdge(v1 : Vertex, v2: Vertex) {
                  var e : Edge = new Edge(v1,v2)
                 edges(ne) = e
                 v1.addEdge(v2)
                 ne = ne + 1
              }
        }

	//var vA : Vertex = new Vertex("a")
	//var vB : Vertex = new Vertex("b")
	//var vC : Vertex = new Vertex("c")
	//var vD : Vertex = new Vertex("d")
	
	//var G : Graph = new Graph()
	
	//G.addVertex(vA)
	//G.addVertex(vB)
	//G.addVertex(vC)
	//G.addVertex(vD)
		
	//G.addEdge(vC,vD)
	//G.addEdge(vA,vB)
    //G.addEdge(vA,vC)	
	//G.addEdge(vB,vD)
        
    var G : Graph = new Graph()
    var v0 : Vertex = new Vertex("0")
    var v1 : Vertex = new Vertex("1")
    var v2 : Vertex = new Vertex("2")
    var v3 : Vertex = new Vertex("3")
    var v4 : Vertex = new Vertex("4")
    var v5 : Vertex = new Vertex("5")
    var v6 : Vertex = new Vertex("6")
    var v7 : Vertex = new Vertex("7")
    
    G.addVertex(v0)
    G.addVertex(v1)
    G.addVertex(v2)
    G.addVertex(v3)
    G.addVertex(v4)
    G.addVertex(v5)
    G.addVertex(v6)
    G.addVertex(v7)
    
    G.addEdge(v6,v4)
    G.addEdge(v2,v0)
    G.addEdge(v0,v1)
    G.addEdge(v7,v6)
    G.addEdge(v0,v5)
    G.addEdge(v0,v7)
    G.addEdge(v1,v3)
    G.addEdge(v5,v6)
    G.addEdge(v6,v2)
    G.addEdge(v4,v0)
    
	// so far, the angels in S are able to do the right thing, no matter in which
	// order we feed the edges. to think about entanglement.
	
	G.checkBipartiteV2(v0)
	G.printBipartite()
     }
}

object Bipartite {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        //val cmdopts = new cli.CliParser(args)
        skalch.AngelicSketchSynthesize(() => 
            new BipartiteSketch())
        }
    }
