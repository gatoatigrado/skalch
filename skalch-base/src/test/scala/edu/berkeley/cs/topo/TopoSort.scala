
package edu.berkeley.cs.topo

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

import scala.collection.mutable.HashSet
import scala.collection.mutable.Set
import scala.collection.mutable.HashMap
import scala.collection.mutable.Map
import scala.collection.mutable.Stack
import scala.collection.mutable.Queue
import scala.collection.mutable.ListBuffer



class TopoSort1Sketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {

        class Vertex(name : String) {
            val MAX_OUT = 5
        
            var id : String = name
            var edges : List[Vertex] = Nil
            
            var sort : Int = -1
        
            override def toString() : String = {
                var returnString : String = id
                return returnString
            }
            
            def toDetailedString() : String = {
                var returnString : String = id
                returnString += "["
                for (j <- 0 to edges.size) returnString += edges(j).id 
                    returnString += "]"
                return returnString
            }
        
            def addEdge(target : Vertex) {
                edges ::= target
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
        
        class Graph {

            var vertices : List[Vertex] = Nil
            var edges : List[Edge] = Nil
            var root : Vertex = null
            
            def addVertex(v : Vertex) { 
                vertices :::= List(v)
                if (root == null) {
                    root = v
                }
            }

            def addEdge(v1 : Vertex, v2: Vertex) {
                var e : Edge = new Edge(v1,v2)
                edges :::= List(e)
                v1.addEdge(v2)
            }
            
            def markNodes() {
                val visited : ListBuffer[Vertex] = new ListBuffer[Vertex]
                visited.append(root)
                val seenEdges : ListBuffer[Edge] = new ListBuffer[Edge]
                for (i <- 0 until edges.size) {
                    val edge : Edge = !!(edges)
                    synthAssert(!seenEdges.contains(edge))
                    synthAssert(visited.contains(edge.src ))
                    
                    seenEdges.append(edge)
                    visited.append(edge.dest)
                    if (!!())
                        setNode(edge.src, visited)
                    if (!!())
                        setNode(edge.dest, visited)
                }
            }
            
            def setNode(v : Vertex, visited : ListBuffer[Vertex]) {
                synthAssert(v.sort == -1)
                v.sort = !!(vertices.size)
            }
            
            def print(ccomponents : List[Set[Vertex]]) {
                var output : String = ""
                ccomponents.foreach { ccomponent =>
                    output += "( "
                    ccomponent.foreach { vertex =>
                        output += vertex.id + " "
                    }
                    output += ") "
                }
                skdprint(output)
            }
            
            def print(ccomponent : Set[Vertex]) {
                var output : String = ""
                output += "( "
                ccomponent.foreach { vertex =>
                    output += vertex.id + " "
                }
                output += ") "
                skdprint(output)
            }
            
            def checkSort(correct : Map[Vertex, Int]) {
                vertices.foreach { v =>
                    val correctSort : Option[Int] = correct.get(v)
                    synthAssert(!correctSort.isEmpty)
                    synthAssert(v.sort == correctSort.get)
//                    synthAssert(v.sort != -1)
//                    v.edges.foreach { dest =>
//                        synthAssert(v.sort > dest.sort)
//                    }
                }
                return;
            }
        }

        var vA : Vertex = new Vertex("a")
        var vB : Vertex = new Vertex("b")
        var vC : Vertex = new Vertex("c")
        var vD : Vertex = new Vertex("d")
        var vE : Vertex = new Vertex("e")
    
        var G : Graph = new Graph()
    
        G.addVertex(vA)
        G.addVertex(vB)
        G.addVertex(vC)
        G.addVertex(vD)
        G.addVertex(vE)
    
        G.addEdge(vA,vB)
        G.addEdge(vA,vC)
        G.addEdge(vB,vD)    
        G.addEdge(vB,vE)
        G.addEdge(vE,vC)
        
        val correct : HashMap[Vertex, Int] = new HashMap[Vertex, Int]
        correct.put(vA, 4)
        correct.put(vB, 3)
        correct.put(vC, 1)
        correct.put(vD, 2)
        correct.put(vE, 2)
        
        G.markNodes()
        G.checkSort(correct)
    }
}

object TopoSort1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        //val cmdopts = new cli.CliParser(args)
        skalch.AngelicSketchSynthesize(() => 
            new TopoSort1Sketch())
        }
    }
