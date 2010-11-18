
package edu.berkeley.cs.bipartite

import scala.collection.immutable.HashMap
import skalch.AngelicSketch
import sketch.util._

class BipartiteClassifier3 extends AngelicSketch {
  val tests = Array(())

  def main() {

    class Vertex(name: String) {
      val MAX_OUT = 5

      var id: String = name
      var edges: Array[Vertex] = new Array[Vertex](MAX_OUT)
      var numedges: Int = 0
      var inset: Int = -1 // uninit
      var visited: Boolean = false

      override def toString(): String = {
        var returnString: String = id
        returnString += "["
        for (j <- 0 to numedges - 1) {
          returnString += edges(j).id
          if (j != numedges - 1) {
            returnString += ","
          }
        }
        returnString += "]"
        return returnString
      }

      def addEdge(target: Vertex) {
        edges(numedges) = target
        numedges += 1
      }
    }

    class Edge(s: Vertex, t: Vertex) {
      var src: Vertex = s
      var dest: Vertex = t
      var handled: Boolean = false

      override def toString(): String = {
        var returnString: String = "<"
        returnString += src
        returnString += ","
        returnString += dest
        returnString += ">"
        return returnString
      }
    }

    def getFunction(): Vertex => Int = {
      val input: List[Int] = List(0, 1)
      val output: List[Int] = List(0, 1)

      var valueMap: Map[Int, Int] = new HashMap[Int, Int]
      for (value <- input) {
        valueMap += value -> !!(output)
      }

      valueMap.foreach(mapping =>
        skdprint(mapping._1 + "-->" + mapping._2))

      var function: Vertex => Int = (vertex => valueMap(vertex.inset))

      return function
    }

    def getFunctionAngelic(): Vertex => Int = {
      var function: Vertex => Int = (vertex => !!(List(0, 1)))

      return function
    }
    
    def getFoldFunction(): (Boolean, Edge) => Boolean = {
      val inputFalse: List[Tuple3[Boolean, Int, Int]] = List((false, 0, 0),
        (false, 0, 1), (false, 1, 0), (false, 1, 1))
      val inputTrue: List[Tuple3[Boolean, Int, Int]] = List((true, 0, 0),
        (true, 0, 1), (true, 1, 0), (true, 1, 1))

      val output: List[Boolean] = List(false, true)

      var valueMap: Map[Tuple3[Boolean, Int, Int], Boolean] =
        new HashMap[Tuple3[Boolean, Int, Int], Boolean]

      // if the input is false, then the output should be false. this is to
      // reduce the state space that needs to be explored
      for (value <- inputFalse) {
        val outputVal: Boolean = !!(output)
        //            val outputVal : Boolean = false
        skdprint(value + "-->" + outputVal)

        valueMap += value -> outputVal
      }
      for (value <- inputTrue) {
        val outputVal: Boolean = !!(output)
        skdprint(value + "-->" + outputVal)

        valueMap += value -> outputVal
      }

      var function: (Boolean, Edge) => Boolean = ((prev, edge) =>
        valueMap((prev, edge.src.inset, edge.dest.inset)))

      return function
    }

    val function : Vertex => Int = getFunction()
    val functionAngelic : Vertex => Int = getFunctionAngelic()
    val foldFunction : (Boolean, Edge) => Boolean = getFoldFunction()
    val init: Boolean = !!()
            
    //	Idea: we will keep marks. we will assume up to two marks can be updated per go. we will force
    //	the angels to reveal to us the order in which edges are picked

    class Graph {
      val MAXV = 10
      val MAXE = 50
      var nv: Int = 0 // number of vertices
      var ne: Int = 0 // number of edges

      var vertices: Array[Vertex] = new Array[Vertex](MAXV)
      var edges: Array[Edge] = new Array[Edge](MAXE)

      // returns a wavefront traversal
      def getEdgeTraversal(root: Vertex): List[Edge] = {
        resetEdgesVertices()
        var edgeOrder: List[Edge] = Nil

        root.visited = true
        for (j <- 0 to ne - 1) { // repeat #edges times
          val e: Edge = edges(!!(ne))
          synthAssert(!e.handled)
          synthAssert(e.src.visited == true)

          // fixed bug? 
          // used to be edgeOrder ::= e
          edgeOrder = edgeOrder ::: List(e)

          skdprint("Selecting Edge " + e)

          e.handled = true
          e.dest.visited = true
        }
        return edgeOrder
      }

      def getEdgeTraversalFixed(): List[Edge] = {
        resetEdgesVertices()
        var edgeOrder: List[Edge] = Nil

        for (j <- 0 to ne - 1) { // repeat #edges times
          val e: Edge = edges(j)

          edgeOrder ::= e
          skdprint("Selecting Edge " + e)
        }
        return edgeOrder
      }

      def classify(root: Vertex): Boolean = {
        skdprint("Traversal")
        val traversal: List[Edge] = getEdgeTraversal(root)

        skdprint(traversal.toString)
        skdprint("Marking")
        root.inset = 0

        val marking: Vertex => Int = function
        //val marking: Vertex => Int = functionAngelic
        for (e <- traversal) {
          val destInset: Int = e.dest.inset
          e.dest.inset = marking(e.src)
        }

        skdprint("Fold traversal")
        val foldTraversal: List[Edge] = getEdgeTraversalFixed()

        skdprint("Folding")
        val fold: (Boolean, Edge) => Boolean = foldFunction
        val foldValue = foldTraversal.foldLeft[Boolean](init)(fold)

        return foldValue
      }

      def printBipartite() = {
        skdprint("Set is:")
        for (j <- 0 to nv - 1) {
          if (vertices(j).inset == 1)
            skdprint(vertices(j).toString)
        }
      }

      def addVertex(v: Vertex) {
        vertices(nv) = v
        nv = nv + 1
      }

      def addEdge(v1: Vertex, v2: Vertex) {
        var e: Edge = new Edge(v1, v2)
        edges(ne) = e
        v1.addEdge(v2)
        ne = ne + 1
      }

      def resetEdgesVertices() {
        for (i <- 0 to nv - 1) {
          vertices(i).visited = false
        }
        for (i <- 0 to ne - 1) {
          edges(i).handled = false
        }
      }
    }

    // bipartite graph
    var v0: Vertex = new Vertex("0")
    var v1: Vertex = new Vertex("1")
    var v2: Vertex = new Vertex("2")
    var v3: Vertex = new Vertex("3")
    var v4: Vertex = new Vertex("4")

    var G: Graph = new Graph()

    G.addVertex(v0)
    G.addVertex(v1)
    G.addVertex(v2)
    G.addVertex(v3)

    G.addEdge(v2, v3)
    G.addEdge(v0, v1)
    G.addEdge(v0, v2)
    G.addEdge(v1, v3)

    synthAssert(G.classify(v0))
    G.printBipartite()

    // non-bipartite graph
    var vA: Vertex = new Vertex("A")
    var vB: Vertex = new Vertex("B")
    var vC: Vertex = new Vertex("C")
    var vD: Vertex = new Vertex("D")
    var vE: Vertex = new Vertex("E")

    var G1 : Graph = new Graph()

    G1.addVertex(vA)
    G1.addVertex(vB)
    G1.addVertex(vC)
    G1.addVertex(vD)

    G1.addEdge(vA, vB)
    G1.addEdge(vA, vC)
    G1.addEdge(vB, vC)
    G1.addEdge(vC, vD)
    G1.addEdge(vD, vB)

    synthAssert(!G1.classify(vA))
    
    // Paper example
    //    var G: Graph = new Graph()
    //    var v0: Vertex = new Vertex("0")
    //    var v1: Vertex = new Vertex("1")
    //    var v2: Vertex = new Vertex("2")
    //    var v3: Vertex = new Vertex("3")
    //    var v4: Vertex = new Vertex("4")
    //    var v5: Vertex = new Vertex("5")
    //    var v6: Vertex = new Vertex("6")
    //    var v7: Vertex = new Vertex("7")
    //
    //    G.addVertex(v0)
    //    G.addVertex(v1)
    //    G.addVertex(v2)
    //    G.addVertex(v3)
    //    G.addVertex(v4)
    //    G.addVertex(v5)
    //    G.addVertex(v6)
    //    G.addVertex(v7)
    //
    //    G.addEdge(v6, v4)
    //    G.addEdge(v2, v0)
    //    G.addEdge(v0, v1)
    //    G.addEdge(v7, v6)
    //    G.addEdge(v0, v5)
    //    G.addEdge(v0, v7)
    //    G.addEdge(v1, v3)
    //    G.addEdge(v5, v6)
    //    G.addEdge(v6, v2)

    // so far, the angels in S are able to do the right thing, no matter in which
    // order we feed the edges. to think about entanglement.
  }
}

object BipartiteRecognizer3 {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    //val cmdopts = new cli.CliParser(args)
    skalch.AngelicSketchSynthesize(() =>
      new BipartiteClassifier3())
  }
}
