package edu.berkeley.cs.mst

import scala.collection.immutable.Map
import scala.collection.immutable.HashMap

import skalch.AngelicSketch

class Mst1Sketch extends AngelicSketch {
  val tests = Array(())

  def main() {
    class Node[T](val elem : T) {
      var edges : List[Tuple2[Node[T], Int]] = Nil
      def addEdge(node : Node[T], weight : Int) {
        edges ::= (node, weight)
      }
    }
    
    class Graph[T] {
      var nodes : List[Node[T]] = Nil
      var edges : List[Tuple3[Node[T],Node[T],Int]] = Nil
    
      def addNode(elem : T) {
        nodes ::= new Node[T](elem)
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
    }
    
    class MST[T](graph : Graph[T]) {
      def getMST() : Graph[T] = {
        val n : Int = graph.nodes.length
        var innode =  Map.empty[Node[T], Boolean]
    
        for (node <- graph.nodes) {
          innode += node -> false
        }
    
        val mst : Graph[T] = new Graph[T]
    
        for (i <- 0 until n) {
          if (i == 0) {
            val node = graph.nodes.head
            innode += (node -> true)
          } else {
            var min : Int = 1000000
            var minEdge : Tuple3[Node[T],Node[T],Int] = null
            for (edge <- graph.edges) {
              skdprint(edge.toString())
              if (innode(edge._1) == true && innode(edge._2) == false) {
                min = Math.min(min, edge._3)
                minEdge = edge
              } else if (innode(edge._1) == false && innode(edge._2) == true) {
                min = Math.min(min, edge._3)
                minEdge = edge
              }
            }
            innode += (minEdge._1 -> true)
            innode += (minEdge._2 -> true)
            mst.addEdge(minEdge._1.elem, minEdge._2.elem, minEdge._3)
          }
        }
        return mst
      }
    }
    
    val g = new Graph[String]
    g.addEdge("a", "b", 4)
    g.addEdge("b", "c", 5)
    g.addEdge("c", "d", 6)
    g.addEdge("a", "d", 7)

    val m = new MST[String](g)
    val mst = m.getMST()
    val edgeSum = mst.edges.foldRight[Int](0)((edge,sum) => edge._3 + sum)
    skdprint(edgeSum.toString())
  }
}

object Mst1 {
  def main(args: Array[String]) = {
    skalch.AngelicSketchSynthesize(() =>
      new Mst1Sketch())
  }
}