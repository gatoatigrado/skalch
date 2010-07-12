
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


class MaxComponent2Sketch extends AngelicSketch {
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
                return returnString
            }
            
            def toDetailedString() : String = {
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
                !!()
                //return ccomponents
                
                val se1 : Set[Edge] = new HashSet[Edge]
                val se2 : Set[Edge] = new HashSet[Edge]
                edges.foreach { edge =>
                    se2.add(edge)
                }
                val sv1 : Set[Vertex] = new HashSet[Vertex]
                val sv2 : Set[Vertex] = new HashSet[Vertex]
                val sv3 : Set[Vertex] = new HashSet[Vertex]
                vertices.foreach { vertex =>
                    sv3.add(vertex)
                }
                
                var i : Int = 10
                while (!sv3.isEmpty && i >= 0) {
                    skdprint("sv1: " + sv1)
                    skdprint("sv2: " + sv2)
                    skdprint("sv3: " + sv3)
                    skdprint("se1: " + se1)
                    skdprint("se2: " + se2)
                    
                    i -= 1
                    val chain : Stack[Set[Vertex]] = new Stack[Set[Vertex]]
                    val v : Vertex = !!(vertices)
                    skdprint("vertex: " + v)
                    synthAssert(sv3.contains(v))
                    sv3.remove(v)
                    sv2.add(v)
                    
                    val singletonSet : Set[Vertex] = new HashSet[Vertex]
                    singletonSet.add(v)
                    chain.push(singletonSet)
                    
                    while (!sv2.isEmpty) {
                        while (!!()) {
                            skdprint("sv2: " + sv2 + "; " + "chain: " + chain)
                            val e : Edge = !!(edges)
                            skdprint("edge: " + e)
                            synthAssert(se2.contains(e))
                            se2.remove(e)
                            se1.add(e)
                            val youngest : Set[Vertex] = chain.top
                            synthAssert(youngest.contains(e.src))
                            
                            if (sv1.contains(e.dest)) {
                                skdprint("case 1")
                                // do nothing
                            } else if (sv2.contains(e.dest)) {
                                skdprint("case 2")
                                val compaction : Set[Vertex] = new HashSet[Vertex]
                                while(!chain.top.contains(e.dest)) {
                                    compaction ++= chain.pop
                                }
                                compaction ++= chain.pop
                                chain.push(compaction)
                            } else if (sv3.contains(e.dest)) {
                                skdprint("case 3")
                                sv3.remove(e.dest)
                                sv2.add(e.dest)
                                
                                val singletonSet : Set[Vertex] = new HashSet[Vertex]
                                singletonSet.add(e.dest)
                                chain.push(singletonSet)
                            }
                        }
                        if (chain.size > 0) {
                            val youngest : Set[Vertex] = chain.pop
                            se2.foreach { edge =>
                                synthAssert(!youngest.contains(edge.src))
                            }
                            ccomponents ::= youngest
                            youngest.foreach { vertex =>
                                sv2.remove(vertex)
                                sv1.add(vertex)
                            }
                        }
                        skdprint("found scc")
                    }
                }
                return ccomponents;
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
                skdprint("All: " + allVertices)
                skdprint("Cur: " + vertices)
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

object MaxComponent2 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        //val cmdopts = new cli.CliParser(args)
        skalch.AngelicSketchSynthesize(() => 
            new MaxComponent2Sketch())
        }
    }
