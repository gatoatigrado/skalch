
package edu.berkeley.cs.maxcomponent

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



class MaxComponent3Sketch extends AngelicSketch {
    val tests = Array( () )
    
    def main() {

        class Vertex(name : String) {
            val MAX_OUT = 5
        
            var id : String = name
            var edges : List[Vertex] = Nil
        
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

    //  Idea: we will keep marks. we will assume up to two marks can be updated per go. we will force
    //  the angels to reveal to us the order in which edges are picked

        class Graph {

            var vertices : List[Vertex] = Nil
            var edges : List[Edge] = Nil

            def addVertex(v : Vertex) { 
                vertices :::= List(v)
            }

            def addEdge(v1 : Vertex, v2: Vertex) {
                var e : Edge = new Edge(v1,v2)
                edges :::= List(e)
                v1.addEdge(v2)
            }
            
            def getConnectedComponents() : List[Set[Vertex]] = {
                var ccomponents : List[Set[Vertex]] = Nil
                val visted : Set[Vertex] = new HashSet[Vertex]
                while (visted.size < vertices.size) {
                    val root : Vertex = !!(vertices)
                    synthAssert(!visted.contains(root))
                    val ccs : List[Set[Vertex]] = search(root, visted)
                    ccomponents :::= ccs
                }
                skdprint("CCs: " + ccomponents)
                return ccomponents;
            }
            
            def search(root : Vertex, visited : Set[Vertex]) : List[Set[Vertex]] = {
                val ccs : ListBuffer[Set[Vertex]] = new ListBuffer[Set[Vertex]]
                dfs(root, ccs, visited)
                skdprint("Search: " + ccs)
                var ccsList : List[Set[Vertex]] = Nil
                ccs.foreach { cc =>
                    ccsList ::= cc
                }
                return ccsList
            }
            
            def dfs(v : Vertex, ccs : ListBuffer[Set[Vertex]], visited : Set[Vertex]) {
                if (visited.contains(v)) {
                    return
                } 
                
                visited.add(v)
                skdprint("Vertex: " + v)
                
                val newSet : Boolean = !!()
                val ccIndex : Int = !!(ccs.size)
                if (newSet) {
                    synthAssert(ccIndex == 0)
                }
                if (newSet) {
                    var newSet : Set[Vertex] = new HashSet[Vertex]
                    newSet.add(v)
                    ccs += newSet
                } else {
                    var component : Set[Vertex] = ccs(ccIndex)
                    component.add(v)
                }
                v.edges.foreach { dest =>
                    skdprint("Edge: " + dest)
                    dfs(dest, ccs, visited)
                }
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
            
            def checkConnectedComponents(ccomponents : List[Set[Vertex]]) {
                val allVertices : Set[Vertex] = new HashSet[Vertex]
                ccomponents.foreach { ccomponent =>
                    synthAssert(isConnected(ccomponent));
                    synthAssert(isMaximal(ccomponent));
                    synthAssert(ccomponent.clone.
                            intersect(allVertices).isEmpty)
                    allVertices ++= ccomponent
                }
                synthAssert(allVertices.size == vertices.size)
            }
            
            def isConnected(ccomponent : Set[Vertex]) : Boolean = {
                ccomponent.foreach { vertex =>
                    var reachable : Set[Vertex] = new HashSet[Vertex]
                    val newVertices : Queue[Vertex] = new Queue[Vertex]
                    newVertices.enqueue(vertex)
                    
                    while(newVertices.size > 0) {
                        val v : Vertex = newVertices.dequeue;
                        reachable.add(v)
                        edges.foreach { edge =>
                            if (edge.src == v && !reachable.contains(edge.dest)) {
                                newVertices.enqueue(edge.dest)
                            }
                        }
                    }
                    val ccomponentClone = ccomponent.clone;
                    if (!ccomponentClone.diff(reachable).isEmpty) {
                        return false
                    }
                }
                return true
            }
            
            def isMaximal(ccomponent : Set[Vertex]) : Boolean = {
                vertices.foreach { vertex =>
                    if (!ccomponent.contains(vertex)) {
                        var maxSet : Set[Vertex] = ccomponent.clone;
                        maxSet.add(vertex);
                        if (isConnected(maxSet)) {
                            return false
                        }
                    }
                }
                return true
            }
        }

        var vA : Vertex = new Vertex("a")
        var vB : Vertex = new Vertex("b")
        var vC : Vertex = new Vertex("c")
        var vD : Vertex = new Vertex("d")
        var vE : Vertex = new Vertex("e")
        var vF : Vertex = new Vertex("f")
    
        var G : Graph = new Graph()
    
        G.addVertex(vA)
        G.addVertex(vB)
        G.addVertex(vC)
        G.addVertex(vD)
        G.addVertex(vE)
        G.addVertex(vF)
    
        G.addEdge(vA,vC)
        G.addEdge(vC,vB)
        G.addEdge(vB,vA)    
        G.addEdge(vB,vD)
        G.addEdge(vD,vA)
        G.addEdge(vE,vB)
        G.addEdge(vF,vE)
        
        val ccomponents : List[Set[Vertex]] = G.getConnectedComponents();
        G.checkConnectedComponents(ccomponents);
    }
}

object MaxComponent3 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        //val cmdopts = new cli.CliParser(args)
        skalch.AngelicSketchSynthesize(() => 
            new MaxComponent3Sketch())
        }
    }
