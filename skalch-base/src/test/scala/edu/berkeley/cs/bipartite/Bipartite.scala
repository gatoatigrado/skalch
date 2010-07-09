
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

          
          def checkBipartite() = {
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

	var vA : Vertex = new Vertex("a")
	var vB : Vertex = new Vertex("b")
	var vC : Vertex = new Vertex("c")
	var vD : Vertex = new Vertex("d")
	
	var G : Graph = new Graph()
	
	G.addVertex(vA)
	G.addVertex(vB)
	G.addVertex(vC)
	G.addVertex(vD)
		
	G.addEdge(vC,vD)
	G.addEdge(vA,vB)
    G.addEdge(vA,vC)	
	G.addEdge(vB,vD)

	// so far, the angels in S are able to do the right thing, no matter in which
	// order we feed the edges. to think about entanglement.
	
	G.checkBipartite()
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
