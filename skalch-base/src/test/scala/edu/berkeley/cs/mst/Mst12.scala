package edu.berkeley.cs.mst

import scala.collection.immutable.Map
import scala.collection.immutable.HashMap

import skalch.AngelicSketch

// n version, updates par to any node, Emina's example (4 nodes, 7 edges),
// order of nodes added follows original algorithm

class Mst12Sketch extends AngelicSketch {
  val tests = Array(())

  def main() {
    class Node[T](val elem : T) {
      var edges : List[Tuple2[Node[T], Int]] = Nil
      def addEdge(node : Node[T], weight : Int) {
        edges ::= (node, weight)
      }
      
      override def toString() : String = {
        return elem.toString()
      }
    }
    
    class Graph[T] {
      var nodes : List[Node[T]] = Nil
      var edges : List[Tuple3[Node[T],Node[T],Int]] = Nil
    
      def addNode(elem : T) {
        nodes ::= new Node[T](elem)
      }
      
      def containsElem(elem : T) : Boolean = {
        val node = nodes.find(node => node.elem == elem).getOrElse(null)
        return node != null
      }
    
      def addEdge(nodeVal1 : T, nodeVal2 : T, weight : Int) {
        var node1 = nodes.find(node => node.elem == nodeVal1).getOrElse(null)
        if (node1 == null) {
          node1 = new Node[T](nodeVal1)
          nodes ::= node1
        }
        var node2 = nodes.find(node => node.elem == nodeVal2).getOrElse(null)
        if (node2 == null) {
          node2 = new Node[T](nodeVal2)
          nodes ::= node2
        }
    
        node1.addEdge(node2, weight)
        node2.addEdge(node1, weight)
        edges ::= (node1, node2, weight)
      }
      
      def getEdge(node1 : Node[T], node2 : Node[T]) : Tuple3[Node[T],Node[T],Int] = {
        val edge = edges.find(edge => (edge._1 == node1 && edge._2 == node2) ||
            (edge._1 == node2 && edge._2 == node1)).getOrElse(null)
        return edge
      }
    }
    
    class MST[T](graph : Graph[T]) {
      def getMST() : Graph[T] = {
        val n : Int = graph.nodes.length
    
        val mst : Graph[T] = new Graph[T]
        var par = Map.empty[Node[T], Node[T]]
        var addedNode : Node[T] = null
        
        // iterate n times
        for (i <- 0 until n) {
          // first iteration
          if (i == 0) {
            // angelically choose the first node
            addedNode = !!(graph.nodes) // graph.nodes.head
            mst.addNode(addedNode.elem)
            skdprint("Added node " + addedNode)
          } else {
            val nodes = graph.nodes
            var min : Int = -1
            var minEdge : Tuple3[Node[T],Node[T],Int] = null
            for (node <- nodes) {
              // find the node with the least distance from (node, mst(node))
              if (!mst.containsElem(node.elem) && par.contains(node)) {
                val edge = graph.getEdge(node, par(node))
                var shouldSwitch = false;
                
                if (edge != null) {
                  if (min == -1) {
                    shouldSwitch = true;
                  } else {
                    shouldSwitch = edge._3 < min
                  }
                } else {
                    synthAssert(false)
                }
                
                if (shouldSwitch) {
                  min = edge._3
                  minEdge = edge
                  addedNode = node
                }
              }
            }
            
            val frontierEdges = graph.edges.filter(edge =>
                (mst.containsElem(edge._1.elem) && !mst.containsElem(edge._2.elem)) ||
                (!mst.containsElem(edge._1.elem) && mst.containsElem(edge._2.elem)))
            
            val minFrontier = frontierEdges.reduceRight((edge1,edge2) => if(edge1._3 > edge2._3) edge2 else edge1)
            
            synthAssert(minFrontier == minEdge)
            
            // add the min edge from (node, par(node))
            mst.addEdge(minEdge._1.elem, minEdge._2.elem, minEdge._3)
            skdprint("Added node " + addedNode)
          }
          // update the nodes not in mst
          val unnodes = graph.nodes.filter(node => !mst.containsElem(node.elem))
          for (unnode <- unnodes) {
            val nodes = graph.nodes
            val nextNode : Int = !!(nodes.length + 2)
            // remove the mapping from par
            if(nextNode == nodes.length + 1) {
              par -= unnode
              skdprint ("Updated node: " + unnode + " -> _")
            } else if(nextNode == nodes.length) {
              skdprint ("Do nothing")  
            } else {
              par += unnode -> nodes(nextNode)
              skdprint ("Updated node: " + unnode + " -> " + nodes(nextNode))
            }
          }
        }
        return mst
      }
    }
    
    val g = new Graph[String]
    g.addNode("n0")
    g.addNode("n1")
    g.addNode("n2")
    g.addNode("n3")
    
    g.addEdge("n0", "n0", 3)
    g.addEdge("n0", "n1", 4)
    g.addEdge("n0", "n3", 2)
    
    g.addEdge("n1", "n2", 0)
    g.addEdge("n1", "n3", 1)
    
    g.addEdge("n2", "n2", 6)
    g.addEdge("n2", "n3", 5)
    
    val m = new MST[String](g)
    val mst = m.getMST()
    val edgeSum = mst.edges.foldRight[Int](0)((edge,sum) => edge._3 + sum)
    synthAssert(edgeSum == 3)
    skdprint(edgeSum.toString())
  }
}

object Mst12 {
  def main(args: Array[String]) = {
    skalch.AngelicSketchSynthesize(() =>
      new Mst12Sketch())
  }
}