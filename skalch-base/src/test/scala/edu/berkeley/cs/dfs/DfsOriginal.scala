// @code standards ignore file
package edu.berkeley.cs.dfs

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
class DfsSketch0() extends AngelicSketch {
  
    val tests = Array( () )

    class Graph(n: Int) {
        val nodes = new ArrayBuffer[Node]

        val e = new Node("e")
        val d = new Node("d")
        val c = new Node("c", e)
        val b = new Node("b", d, e)
        val a = new Node("a", b, c)
        
        val root = a

        nodes ++= List(a,b,c,d,e)

        def checkpoint() {
            for(node <- nodes) {
                node.checkpoint()
            }
        }

        def unchanged(): Boolean = {
            for(node <- nodes if node.changed) {
                return false
            }

            true
        }

        override def toString(): String = {
            var string = ""

            for(node <- nodes) {
                string += node.name + ": ["
                for(child <- node.children) {
                    string += child.name + " "
                }
                string += "]   "
            }

            string
        }
    }

    trait LocationLender[A] {
        def locations():Seq[Location[A]]
    }

    class Node(val name: String, newChildren: Node*) extends LocationLender[Node] {
        var children = new ArrayBuffer[Node]()
        var checkpointChildren: Buffer[Node] = null

        var visited  = false

        for(child <- newChildren) children += child

        def visit() {
            synthAssert(visited == false)
            this.visited = true
        }

        def checkpoint() {
            checkpointChildren = children.clone
        }

        def changed() = !unchanged

        def unchanged(): Boolean = {
            if(children.length != checkpointChildren.length) {
                return false
            } else {
                for((a, b) <- checkpointChildren.zip(children)) {
                    if(a != b) {
                        return false
                    }
                }
            }

            true
        }

        override def toString(): String = {
            name
        }

        def locations() = {
            mkLocations(children)
        }
    }

    abstract class Location[A] {
        def read(): A
        def write(x: A)
        def == (that: Location[A]) = false
    }

    class GeneralLocation[A](val reader: () => A, val writer: (A) => Unit) extends Location[A] {
        def read() = {
            reader()
        }

        def write(x: A) {
            writer(x)
        }
    }

    class BufferLocation[A](val buffer: Buffer[A], val index: Int) extends Location[A] {
        override def read() = {
            buffer(index)
        }

        override def write(x: A) {
            buffer(index) = x
        }

        override def == (that: Location[A]) = {
            that.isInstanceOf[BufferLocation[A]] &&
            buffer == that.asInstanceOf[BufferLocation[A]].buffer && index == that.asInstanceOf[BufferLocation[A]].index
        }
    }

    class KeyholeStack[A <: LocationLender[A]](allowedExtraStorage: Int = 0, domainA:Seq[A]) {
        val reference = new Stack[A]            // reference stack
        val locations = new Stack[Location[A]]  // where the actual values are stored

        val extraStorage = new ArrayBuffer[A]

        var i = 0
        while(i < allowedExtraStorage) {
            extraStorage += domainA(0) //!!(domainA)
            i += 1
        }

        val borrowed = new HashMap[A, Int]

        val popp1 = 0 //!!(List(0, 1, 2))
        val popp2 = 0 //!!(List(0, 1))

        val pushp1 = 2 //!!(List(0, 1, 2))
        val pushp2 = 1 //!!(List(0, 1))

        val extraLocations = mkLocations(extraStorage)

        def push(x: A, have: A) {
            val to = x.locations
            reference.push(x)

            var i = 0
            while(i < to.length) {
                if(to(i).read == have) {
                    borrowed(x) = i
                    i += 10000 // my kingdom for a break statement
                }
                i += 1
            }

            val borrowedLoc = to(borrowed(x))

            if(have != null) {
                synthAssert(borrowedLoc.read == have)
            }

            locations.push(borrowedLoc)

            val borrowedVal = borrowedLoc.read  // this value must be saved

            var allLocations = List(borrowedLoc) ++ extraLocations

            val values = new ListBuffer[A]

            values += borrowedVal
            values += x

            skdprint(extraStorage.mkString("push("+x.toString()+") : (", ", ", ")") + " borrowedVal:[" + borrowedVal.toString() + "]")

            val permutation = new Stack[Int]
            for(location <- extraLocations) {
                values += location.read
            }

            borrowedLoc.write(values.remove(pushp1))
            extraLocations(0).write(values.remove(pushp2))

            synthAssert(x == extraLocations(0).read)

            skdprint(extraStorage.mkString("push("+x.toString()+") : (", ", ", ")") + " borrowedVal:[" + borrowedVal.toString() + "]")
        }

        def pop(restore: List[A]) = {
            synthAssert(reference.size > 0)
            val refPoppedVal = reference.pop
            synthAssert(borrowed.contains(extraLocations(0).read))
            val borrowedLoc = extraLocations(0).read.locations.apply(borrowed(extraLocations(0).read))

            var allLocations = List(borrowedLoc) ++ extraLocations
            val values = new ListBuffer[A]
            var found = false
            values ++= restore
            for(location <- allLocations) {
                val v = location.read
                values += v
                if(refPoppedVal == v)
                    found = true
            }
            synthAssert(found)

            skdprint(extraStorage.mkString("pop: (", ", ", ")") + " borrowdVal:[" + borrowedLoc.read.toString() + "] returns:" + refPoppedVal.toString())

            borrowedLoc.write(values.remove(popp1))
            extraLocations(0).write(values.remove(popp2))

            skdprint(extraStorage.mkString("pop: (", ", ", ")") + " borrowdVal:[" + borrowedLoc.read.toString() + "] returns:" + refPoppedVal.toString())

            refPoppedVal
        }
    }

    def mkLocations[A](b: Buffer[A]): Seq[Location[A]] = {
        val locations = new ArrayBuffer[Location[A]]

        var i = 0
        while(i < b.length) {
            locations += new BufferLocation(b, i)
            i += 1
        }

        locations
    }

    def dfs(g: Graph) {
        val root   = g.root
        val origin = new Node("origin", root)

        val stack = new KeyholeStack[Node](1, g.nodes)
        stack.push(origin, root)

        synthAssert(origin.children(0)==root)

        var current  = root

        var step = 0
        while(current != origin) {
            synthAssert(step < 10)
            step += 1

            if(current.visited) {
                skdprint("Backtracking to: " + current.name)
            } else {
                skdprint("Visiting: " + current.name)
                current.visit()
            }

            var next: Node = null

            for(child <- current.children if child != null && !child.visited && next == null) {
                next = child
            }

            if(next != null) {
                stack.push(current, next)
                skdprint("graph after push:" + g.toString())
                current = next
            } else {
                // backtrack
                current = stack.pop(List(current))
                skdprint("graph after pop:" + g.toString())
                
            }
        }
    }
    
    def main() = {
        val g = new Graph(3)
        g.checkpoint
        
        dfs(g)
        
        synthAssert(g.unchanged)
        
        for(node <- g.nodes) {
            synthAssert(node.visited)
        }
    }
}

object Dfs0 {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => new DfsSketch0())
    }
}

