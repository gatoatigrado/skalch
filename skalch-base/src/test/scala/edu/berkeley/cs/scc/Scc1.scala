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
 * Non-working version
 */
class SccSketch1() extends AngelicSketch {
    val tests = Array( () )

    def main() : Unit = {
        class Vertex(name : String) {
            var id : String = name
            var edges : List[Vertex] = Nil
        
            override def toString() : String = {
                return id
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

            var scc : ListBuffer[Set[Vertex]] = new ListBuffer[Set[Vertex]]
            
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
            
            def getSCC() : Set[Set[Vertex]] = {
                var root : Vertex = vertices.head
                var visitedNodes : ListBuffer[Vertex] = new ListBuffer[Vertex]();
                var i : Int = 10
                visitNode(root, 0)
                
                var sccSet : Set[Set[Vertex]] = new HashSet[Set[Vertex]]
                for (cc <- scc) {
                    sccSet.add(cc);
                }
                return sccSet;
            }
            
            def visitNode(node : Vertex, depth : Int) {
                if (depth > 10) {
                    return
                }
                val choice : Int = !!(List(1,2,3,4))
                
                choice match {
                    case 1 => return;
                    case 2 => !!(scc).add(node)
                    case 3 => val newSet : Set[Vertex] = new HashSet[Vertex]
                         newSet.add(node)
                         scc += newSet
                    case _ => ()
                }
                
                for (child <-node.edges) {
                    visitNode(child, depth + 1)
                }
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
        
        graph.addEdge(v1, v2)
        graph.addEdge(v1, v4)
        graph.addEdge(v2, v3)
        graph.addEdge(v3, v1)
        graph.addEdge(v3, v4)
        graph.addEdge(v4, v5)
        
        val scc : Set[Set[Vertex]] = graph.getSCC()
        
        val scc1 : Set[Vertex] = new HashSet[Vertex]()
        scc1.add(v1)
        scc1.add(v2)
        scc1.add(v3)
        
        val scc2 : Set[Vertex] = new HashSet[Vertex]()
        scc1.add(v3)
        
        val scc3 : Set[Vertex] = new HashSet[Vertex]()
        scc1.add(v4)
        
        val sccSolution : Set[Set[Vertex]] = new HashSet[Set[Vertex]]()
        sccSolution.add(scc1)
        sccSolution.add(scc2)
        sccSolution.add(scc3)
        
        synthAssert(sccSolution.equals(scc))
    }
}

object Scc1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => 
            new SccSketch1())
    }
}