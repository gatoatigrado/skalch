package edu.berkeley.cs.scc

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util._

import scala.collection.mutable.HashSet
import scala.collection.mutable.Set
import scala.collection.mutable.HashMap
import scala.collection.mutable.Map
import scala.collection.mutable.Stack
import scala.collection.mutable.Queue
import scala.collection.mutable.ListBuffer

/*
 * Does DFS traversal of graph. Angelically figures out where to merge. Aligns angels.
 */
class SccSketch5() extends AngelicSketch {
    val tests = Array( () )

    def main() : Unit = {
        class Vertex(private val id : String) {
            private var connectedVertices : List[Vertex] = Nil
            private var representative : Vertex = this
            private var numChanged : Int = 0
        
            override def toString() : String = {
                return id
            }
            
            def toDetailedString() : String = {
                var returnString : String = id
                returnString += "["
                for (j <- 0 to connectedVertices.size) returnString += connectedVertices(j).id 
                    returnString += "]"
                return returnString
            }
        
            def addEdge(target : Vertex) {
                connectedVertices ::= target
            }
            
            def getEdges() : List[Edge] = {
                connectedVertices.map(vertex => new Edge(this, vertex))
            }
            
            def getConnectedVertices() : List[Vertex] = {
                return connectedVertices
            }
            
            def getRepresentative() : Vertex = {
                representative
            }
            
            def setRepresentative(r : Vertex) {
                if (numChanged == 0) {
                    representative = r
                    numChanged += 1
                } else {
                    synthAssert(false)
                }
            }
            
            override def equals(other : Any) : Boolean = {
                val otherRef : AnyRef = other.asInstanceOf[AnyRef]
                if (otherRef == null) {
                    return false
                }
                return this.eq(otherRef)
            }
        }

        class Edge(val src : Vertex, val dest : Vertex) {

            override def toString() : String = {
                var returnString : String = "<"
                returnString += src
                returnString += ","
                returnString += dest
                returnString += ">"
                return returnString
            }
            
            override def equals(other : Any) : Boolean = {
                val otherEdge : Edge = other.asInstanceOf[Edge]
                if (otherEdge == null) {
                    return false
                }
                return src.equals(otherEdge.src) && dest.equals(otherEdge.dest)
            }
        }
        
        class VertexTraversal(forward : Boolean, vertex : Vertex) {
            var availableVertices : List[Vertex] = List(vertex) 
            var visitOrder : ListBuffer[Vertex] = new ListBuffer[Vertex]
            
            determineTraversal()
            
            def getVertex() : Vertex = {
                if (forward) {
                    return visitOrder.remove(0)
                } else {
                    return visitOrder.remove(visitOrder.size - 1)
                }
            }
            
            private def next() : Vertex = {
                if (availableVertices.isEmpty) {
                     return null
                }
                
                val nextVertexIndex : Int = !!(availableVertices.size)
                skput_check(nextVertexIndex)
                val nextVertex : Vertex = availableVertices.drop(nextVertexIndex).head
                availableVertices -= nextVertex
                for (vertex <- nextVertex.getConnectedVertices) {
                    if (!visitOrder.contains(vertex)) {
                        availableVertices ::= vertex
                    }
                }
                visitOrder += nextVertex
                return nextVertex
            }
            
            private def determineTraversal() {
                while (next() != null) {}
                skdprint(visitOrder.toString)
            }
        }
        
        class EdgeTraversal(forward : Boolean, vertex : Vertex) {
            var availableEdges : List[Edge] = vertex.getEdges() 
            var visitOrder : ListBuffer[Edge] = new ListBuffer[Edge]
            
            determineTraversal()
            
            def getEdge() : Edge = {
                if (visitOrder.isEmpty) {
                    return null
                } else if (forward) {
                    return visitOrder.remove(0)
                } else {
                    return visitOrder.remove(visitOrder.size - 1)
                }
            }
            
            private def next() : Edge = {
                if (availableEdges.isEmpty) {
                     return null
                }
                
                //val nextEdgeIndex : Int = availableEdges.size - 1
                val nextEdgeIndex : Int = 0
                
                skput_check(nextEdgeIndex)
                val nextEdge : Edge = availableEdges.drop(nextEdgeIndex).head
                
                availableEdges -= nextEdge
                for (edge <- nextEdge.dest.getEdges()) {
                    if (!visitOrder.contains(edge)) {
                        availableEdges ::= edge
                    }
                }
                //skdprint(availableEdges.toString)
                visitOrder += nextEdge
                return nextEdge
            }
            
            private def determineTraversal() {
                while (next() != null) {}
                skdprint(visitOrder.toString)
            }
        }
        
        class Graph {

            var vertices : List[Vertex] = Nil
            var edges : List[Edge] = Nil
            var root : Vertex = null
            
            val sccs : HashMap[Vertex, Int] = new HashMap[Vertex, Int] 
            var curVal : Int = 0
            
            var scc : ListBuffer[Set[Vertex]] = new ListBuffer[Set[Vertex]]
            
            def addVertex(v : Vertex) { 
                vertices ::= v
                if (root == null) {
                    root = v
                }
            }

            def addEdge(v1 : Vertex, v2 : Vertex) {
                var e : Edge = new Edge(v1,v2)
                edges ::= e
                v1.addEdge(v2)
            }
            
            def mark() {
                skdprint("Marking graph nodes")
                val traversal = new EdgeTraversal(!!(), root)
                var edge : Edge = traversal.getEdge()
                
                while (edge != null) {
                    skdprint("Visiting edge: " + edge)
                    val shouldMerge : Boolean = !!()
                    skput_check(shouldMerge)
                    val useSrc : Boolean = !!()
                    if (shouldMerge) {
                        merge(useSrc, edge.src, edge.dest)
                    }
                    edge = traversal.getEdge()
                }
            }
            
            def merge(changeV1 : Boolean, v1 : Vertex, v2 : Vertex) {
                skdprint("Merging: (" + v1 + ", " + v2 + ")")
                skput_check(changeV1)
                var changedVertex : Vertex = null 
                if (changeV1) {
                    skdprint("Using: " + v2.getRepresentative)
                    v1.setRepresentative(v2.getRepresentative)
                } else {
                    skdprint("Using: " + v1.getRepresentative)
                    v2.setRepresentative(v1.getRepresentative)
                }
            }
            
            def isSameClass(v1 : Vertex, v2 : Vertex) : Boolean = {
                v1.getRepresentative() == v2.getRepresentative()
            }
        }
        
        val graph : Graph = new Graph()
        
        
        val v1 : Vertex = new Vertex("v1")
        graph.addVertex(v1)
        val v2 : Vertex = new Vertex("v2")
        graph.addVertex(v2)
        val v3 : Vertex = new Vertex("v3")
        graph.addVertex(v3)
        val v4 : Vertex = new Vertex("v4")
        graph.addVertex(v4)
        val v5 : Vertex = new Vertex("v5")
        graph.addVertex(v5)
        val v6 : Vertex = new Vertex("v6")
        graph.addVertex(v6)
        
        
        graph.addEdge(v1, v2)
        graph.addEdge(v2, v3)
        graph.addEdge(v3, v4)
        graph.addEdge(v4, v1)
        graph.addEdge(v2, v5)
        graph.addEdge(v5, v4)
        graph.addEdge(v5, v6)
        
        
        graph.mark()
        
        synthAssert(graph.isSameClass(v1, v2) == true)
        synthAssert(graph.isSameClass(v1, v3) == true)
        synthAssert(graph.isSameClass(v1, v4) == true)
        synthAssert(graph.isSameClass(v1, v5) == true)
        
        
        synthAssert(graph.isSameClass(v1, v6) == false)
    }
}

object Scc5 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new SccSketch5())
    }
}