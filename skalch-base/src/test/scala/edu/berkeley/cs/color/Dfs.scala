// @code standards ignore file
package edu.berkeley.cs.color

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

import scala.collection.mutable.Stack
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.Buffer
import scala.collection.mutable.ListBuffer

/** @author Casey Rodarmor */
class DfsSketch() extends AngelicSketch {

  val tests = Array(())

  class Graph() {
    var nodes = new ArrayBuffer[Node]

    val d = new Node("d")
    val c = new Node("c", d)
    val b = new Node("b")
    val a = new Node("a", b, c)

    val root = a

    nodes ++= List(a, b, c, d)

    def checkpoint() {
      for (node <- getNodes()) {
        node.checkpoint()
      }
    }

    def getNodes(): ArrayBuffer[Node] = {
      return nodes;
    }

    def unchanged(): Boolean = {
      for (node <- getNodes() if node.changed) {
        return false
      }

      true
    }

    override def toString(): String = {
      var string = ""

      for (node <- getNodes()) {
        string += node.toStringChild()
      }

      string
    }
  }

  trait LocationLender[A] {
    def locations(): Seq[Location[A]]
  }

  class Node(val name: String, newChildren: Node*) extends LocationLender[Node] {
    var children = new ArrayBuffer[Node]()
    var checkpointChildren: Buffer[Node] = null

    var visited = false

    for (child <- newChildren) children += child

    def visit() {
      synthAssert(visited == false)
      visited = true
    }

    def checkpoint() {
      checkpointChildren = children.clone
    }

    def changed() = !unchanged

    def unchanged(): Boolean = {
      if (children.length != checkpointChildren.length) {
        return false
      } else {
        for ((a, b) <- checkpointChildren.zip(children)) {
          if (a != b) {
            return false
          }
        }
      }

      true
    }

    override def toString(): String = {
      name
    }

    def toStringChild(): String = {
      var string = name + ":["
      for (child <- children) {
        string += child.name + " "
      }
      string += "] "
      return string
    }

    def locations() = {
      mkLocations(children)
    }
  }

  abstract class Location[A] {
    def read(): A
    def write(x: A)
    def ==(that: Location[A]) = false
  }

  class BufferLocation[A](val buffer: Buffer[A], val index: Int) extends Location[A] {
    override def read() = {
      buffer(index)
    }

    override def write(x: A) {
      buffer(index) = x
    }

    override def ==(that: Location[A]) = {
      that.isInstanceOf[BufferLocation[A]] &&
        buffer == that.asInstanceOf[BufferLocation[A]].buffer && index == that.asInstanceOf[BufferLocation[A]].index
    }
  }

  class KeyholeStack[A <: LocationLender[A]](allowedExtraStorage: Int = 0, domainA: Seq[A]) {
    // reference stack, should only be used to compare values returned by KeyholeStack
    val reference = new Stack[A]

    // limited storage (fixed number of cells)
    val extraStorage = new ArrayBuffer[A]

    var i = 0
    while (i < allowedExtraStorage) {
      extraStorage += !!(domainA)
      println("Chose extra: " + extraStorage(extraStorage.size - 1), sklast_angel_color());
      i += 1
    }

    val extraLocations = mkLocations(extraStorage)

    def getRef(): Stack[A] = {
      return reference;
    }

    def push(current: A, next: A) {
      // used to limit the search path of the synthesizer
      reference.push(current)

      val nodeList = List(current, next)
      val nodeIndex = !!(nodeList.size)
      skput_check(nodeIndex)
      val node = nodeList(nodeIndex)
      if (nodeIndex == 0) {
          println("Chose current: " + node, sklast_angel_color());
      } else {
          println("Chose next: " + node, sklast_angel_color());    
      }
      
      val borrowedCurIndex = !!(node.locations().size)
      skput_check(borrowedCurIndex)
      val borrowedCur = node.locations()(borrowedCurIndex)
      println("Chose borrowed node: " + borrowedCur.read(), sklast_angel_color());
      
      var allLocations = List(borrowedCur) ++ extraLocations

      val values = new ListBuffer[A]

      // must save value in borrowedLoc
      values += borrowedCur.read()
      values += current
      values += next
      for (location <- extraLocations) {
        values += location.read()
      }

      borrowedCur.write(values.remove(skput_check(!!(values.size))))
      println("Save into borrowed: " + borrowedCur.read(), sklast_angel_color())
      for (location <- extraLocations) {
        location.write(values.remove(skput_check(!!(values.size))))
        println("Save into extra: " + location.read(), sklast_angel_color());
      }
    }

    def pop(restore: A) = {
      synthAssert(reference.size > 0)
      // should only be used to check if the correct value is returned
      val refPoppedVal = reference.pop

      var nodes = new ListBuffer[A]
      nodes += restore

      for (location <- extraLocations) {
        nodes += location.read()
      }

      var nodeIndex = !!(nodes.size)
      println("Node index", sklast_angel_color());
      skput_check(nodeIndex)
      var node = nodes(nodeIndex)

      var borrowedLocIndex = !!(node.locations().size)
      println("Node", sklast_angel_color());
      skput_check(borrowedLocIndex)
      var borrowedLoc = node.locations()(borrowedLocIndex)

      var allLocations = List(borrowedLoc) ++ extraLocations

      val values = new ListBuffer[A]

      for (location <- allLocations) {
        val v = location.read
        values += v
      }
      values += restore

      val returnValue = !!(values)
      synthAssert(returnValue == refPoppedVal)

      println("pop(" + restore.toString() + ") : "
        + extraStorage.mkString("e[", ", ", "]")
        + " return[" + refPoppedVal.toString() + "]")

      for (location <- allLocations) {
        location.write(values.remove(skput_check(!!(values.size))))
        println("Write", sklast_angel_color());
      }

      returnValue
    }
  }

  def mkLocations[A](b: Buffer[A]): Seq[Location[A]] = {
    val locations = new ArrayBuffer[Location[A]]

    var i = 0
    while (i < b.length) {
      locations += new BufferLocation(b, i)
      i += 1
    }

    locations
  }

  def dfs(g: Graph) {
    val root = g.root
    val origin = new Node("origin", root)

    val stack = new KeyholeStack[Node](1, g.nodes)
    stack.push(origin, root)

    var current = root

    var step = 0
    println("original graph: " + origin.toStringChild() + g.toString() + "ex:"
      + stack.extraStorage.mkString("[", ",", "]"))

    while (current != origin) {
      synthAssert(step < 10)
      step += 1

      if (current.visited) {
        println("Backtracking: " + current.name)
      } else {
        println("Visiting: " + current.name)
        current.visit()
      }

      var next: Node = null

      for (child <- current.children if child != null && !child.visited && next == null) {
        next = child
      }

      if (next != null) {
        stack.push(current, next)

        println("graph after push: " + origin.toStringChild() + g.toString() + "ex:"
          + stack.extraStorage.mkString("[", ",", "]"))
        current = next
      } else {
        // backtrack
        current = stack.pop(current)
        println("graph after pop: " + origin.toStringChild() + g.toString() + "ex:"
          + stack.extraStorage.mkString("[", ",", "]"))
      }
    }

    synthAssert(stack.getRef().isEmpty)
  }

  def toString(nodes: Buffer[Node]): String = {
    var string = ""

    for (node <- nodes) {
      string += node.name + ": ["
      for (child <- node.children) {
        string += child.name + " "
      }
      string += "]   "
    }

    string
  }

  def main() = {
    val g = new Graph()
    g.checkpoint

    dfs(g)

    synthAssert(g.unchanged)

    for (node <- g.nodes) {
      synthAssert(node.visited)
    }
    println("Final graph " + g.toString())
  }
}

object Dfs {
  def main(args: Array[String]) = {
    val cmdopts = new cli.CliParser(args)
    BackendOptions.addOpts(cmdopts)
    skalch.AngelicSketchSynthesize(() => new DfsSketch())
  }
}

