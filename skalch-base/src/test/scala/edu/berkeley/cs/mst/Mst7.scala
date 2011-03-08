package edu.berkeley.cs.mst

import scala.collection.immutable.Map
import scala.collection.immutable.HashMap

import skalch.AngelicSketch

// n version,  updates par to the newly added node, Emina's example (9 nodes, 15 edges)

class Mst7Sketch extends AngelicSketch {
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
        
        for (i <- 0 until n) {
          if (i == 0) {
            addedNode = !!(graph.nodes) // graph.nodes.head
            mst.addNode(addedNode.elem)
            skdprint("Added node " + addedNode)
          } else {
            val nodes = graph.nodes
            var min : Int = -1
            var minEdge : Tuple3[Node[T],Node[T],Int] = null
            for (node <- nodes) {
              if (!mst.containsElem(node.elem) && par.contains(node)) {
                val edge = graph.getEdge(node, par(node))
                var shouldSwitch = false;
                if (edge != null) {
                  if (min == -1) {
                    shouldSwitch = true;
                  } else {
                    shouldSwitch = edge._3 < min
                  }
                }
                if (shouldSwitch) {
                  min = edge._3
                  minEdge = edge
                  addedNode = node
                }
              }
            }
            mst.addEdge(minEdge._1.elem, minEdge._2.elem, minEdge._3)
            skdprint("Added node " + addedNode)
          }
          val unnodes = graph.nodes.filter(node => !mst.containsElem(node.elem))
          for (unnode <- unnodes) {
            if(!!()) {
              par += unnode -> addedNode
              skdprint ("Updated node: " + unnode + " -> " + addedNode)
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
    g.addNode("n4")
    g.addNode("n5")
    g.addNode("n6")
    g.addNode("n7")
    g.addNode("n8")
    
    g.addEdge("n0", "n2", 12)
    g.addEdge("n0", "n8", 1)
    g.addEdge("n0", "n3", 13)
    
    g.addEdge("n1", "n8", 5)
    g.addEdge("n1", "n7", 8)
    g.addEdge("n1", "n2", 9)
    
    g.addEdge("n2", "n8", 11)
    g.addEdge("n2", "n7", 4)
    g.addEdge("n2", "n3", 10)
    
    g.addEdge("n3", "n7",2)
    
    g.addEdge("n4", "n7", 6)
    g.addEdge("n4", "n8", 14)
    
    g.addEdge("n5", "n6", 3)
    
    g.addEdge("n6", "n8", 7)
    
    g.addEdge("n7", "n8", 0)
    
    val m = new MST[String](g)
    val mst = m.getMST()
    val edgeSum = mst.edges.foldRight[Int](0)((edge,sum) => edge._3 + sum)
    synthAssert(edgeSum == 28)
    skdprint(edgeSum.toString())
  }
}

object Mst7 {
  def main(args: Array[String]) = {
    skalch.AngelicSketchSynthesize(() =>
      new Mst7Sketch())
  }
}