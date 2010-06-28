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
class DfsSketch1() extends AngelicSketch {
  
    val tests = Array( () )

    class Graph() {
        val nodes = new ArrayBuffer[Node]

        val d = new Node("d")
        val c = new Node("c", d)
        val b = new Node("b")
        val a = new Node("a", b, c)

        val root = a

        nodes ++= List(a,b,c,d)

        def checkpoint() {
            for(node <- getNodes()) {
                node.checkpoint()
            }
        }
        
        def getNodes() : ArrayBuffer[Node] = {
          return nodes;
        }

        def unchanged(): Boolean = {
            for(node <- getNodes() if node.changed) {
                return false
            }

            true
        }

        override def toString(): String = {
            var string = ""

            for(node <- getNodes()) {
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
        // reference stack, should only be used to compare values returned by KeyholeStack
        val reference = new Stack[A]

        // limited storage (fixed number of cells)
        val extraStorage = new ArrayBuffer[A]

        var i = 0
        while(i < allowedExtraStorage) {
            extraStorage += !!(domainA)
            i += 1
        }
        
        val extraLocations = mkLocations(extraStorage)

        def push(current: A, next: A) {
            // used to limit the search path of the synthesizer
            reference.push(current)

            // find pointer from current to next
            val to = current.locations

            var i = 0
            var borrowedIndex = -1
            while(i < to.length) {
                if(to(i).read == next) {
                    borrowedIndex = i
                    i = to.length // my kingdom for a break statement
                }
                i += 1
            }

            val borrowedCur = to(borrowedIndex)
            if(next != null) {
                synthAssert(borrowedCur.read == next)
            }
            
            var allLocations = List(borrowedCur) ++ extraLocations

            val values = new ListBuffer[A]

            // must save value in borrowedLoc
            values += borrowedCur.read
            values += current
            values += next
            for(location <- extraLocations) {
                values += location.read
            }

            skdprint(extraStorage.mkString("push("+current.toString()+") : (", ", ", ")") +
                       " borrowedVal:[" + borrowedCur.read.toString() + "]")

            borrowedCur.write(values.remove(!!(values.size)))
            for(location <- extraLocations) {
                location.write(values.remove(!!(values.size)))
            }

            skdprint(extraStorage.mkString("push("+current.toString()+") : (", ", ", ")") +
                       " borrowedVal:[" + borrowedCur.read.toString() + "]")
        }

        def pop(restore: A) = {
            synthAssert(reference.size > 0)
            // should only be used to check if the correct value is returned
            val refPoppedVal = reference.pop
            
            var allLocations = extraLocations
            var childLocations = new ArrayBuffer[Location[A]]
            for (location <-allLocations) {
              childLocations ++= location.read().locations()
            }
            allLocations ++= childLocations
            
            allLocations ++= restore.locations()

            val values = new ListBuffer[A]
            
            for(location <- allLocations) {
                val v = location.read
                values += v
            }
            values += restore
            
            val returnValue = !!(values)
            synthAssert(returnValue == refPoppedVal)
            
      /*      
            allLocations ++= returnValue.locations()
            for(location <- returnValue.locations()) {
                val v = location.read
                values += v
            }
        */    
            
            skdprint(extraStorage.mkString("pop: (", ", ", ")") + " returns:" + refPoppedVal.toString())

            for(location <- allLocations) {
                location.write(values.remove(!!(values.size)))
            }

            skdprint(extraStorage.mkString("pop: (", ", ", ")") + " returns:" + refPoppedVal.toString())

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

        var current = root

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
                skdprint("graph before pop:" + g.toString())
                current = stack.pop(current)
            }
        }
    }
    
    def main() = {
        val g = new Graph()
        skdprint("Original graph " + g.toString())
        g.checkpoint
        
        dfs(g)
        
        synthAssert(g.unchanged)
        
        for(node <- g.nodes) {
            synthAssert(node.visited)
        }
        skdprint("Final graph " + g.toString())
    }
}

object Dfs1 {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => new DfsSketch1())
    }
}

