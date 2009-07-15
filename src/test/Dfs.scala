package test
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

import scala.collection.mutable.Stack
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Buffer
import scala.collection.mutable.ListBuffer
import scala.runtime.RichInt

class DfsSketch() extends DynamicSketch {
    def dysketch_main() = {
        val g = new Graph(3)
        g.checkpoint

        dfs(g)

        synthAssertTerminal(g.unchanged)

        for(node <- g.nodes) {
            synthAssertTerminal(node.visited)
        }

        true
    }

    /*
    each child with multiple children gives the oracle to re-order those children, and the fix them sometime on the way up, giving more valid traces

    a node with two children that point to the same node will also introduce new multiple traces
    */

    class Graph(n: Int, seed: Long = 1983) {
        val nodes = new ArrayBuffer[Node]


        val e = new Node("e")
        val d = new Node("d", e)
        val c = new Node("c", d, e)
        val b = new Node("b", c, e)
        val a = new Node("a", b)
        e.children += a
        d.children += b

/*
        val e = new Node("e")
        val d = new Node("d", e)
        val c = new Node("c", d, e)
        val b = new Node("b", c, e)
        val a = new Node("a", b)
*/    
        val root = a

        nodes ++= List(a,b,c,d,e)
        
        /* // code for randomly generating a graph
        {
            var i: Int = 0

            import scala.util.Random
          
            i = 0
            while(i < n) {
                nodes += new Node("" + i)
                i += 1
            }

            val rng = new Random
            rng.setSeed(seed)

            i = 0
            while(i < n * 10) {
                val parent = nodes(rng.nextInt(nodes.length))
                val child  = nodes(rng.nextInt(nodes.length))
                parent.children += child
                i += 1
            }

            nodes(0)
        }
        */

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
            synthAssertTerminal(visited == false)
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

    class AngelicStack[A]() {
        val cheat = new ArrayBuffer[A]

        def push(x: A) {
            cheat += x
        }

        def pop() = {
            cheat.remove(!!(cheat.length))
        }
    }

    class KeyholeStack[A <: LocationLender[A]](allowedExtraStorage: Int = 0, domainA:Seq[A]) {
        val reference = new Stack[A]            // reference stack
        val locations = new Stack[Location[A]]  // where the actual values are stored

        val extraStorage = new ArrayBuffer[A]

        for (i <- 1 to allowedExtraStorage) {
            extraStorage += !!(domainA)
        }
        
        val popp1 = !!(0 until 3)
        val popp2 = !!(0 until 2)

        val pushp1 = !!(0 until 3)
        val pushp2 = !!(0 until 2)
//        val pushp3 = !!(0 until 1)
        
        val extraLocations = mkLocations(extraStorage)

        def push(x: A, to: Seq[Location[A]]) {
            reference.push(x)

            val borrowedLoc = !!(to)
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
            // for (i <- 1 to values.length) {
            //    // generate the last index first
            //    permutation.push(!!(i))
            // }
            //for(location <- allLocations) {
            //      location.write(values.remove(!!(values.length)))
                  // location.wite(values.remove(permutation.pop))
            //}
            
            borrowedLoc.write(values.remove(pushp1))//!!(values.length))) 
            extraLocations(0).write(values.remove(pushp2))//!!(values.length))) 
            // extraLocations(1).write(values.remove(!!(values.length))) 

            //- borrowedLoc.write(x)  // mistake!
            //- extraLocations(0).write(borrowedVal)
            
            // an attempt to determinize (quite successful on a chain graph)
            // store the value from borrowed location into the local location
            //- synthAssertTerminal(borrowedVal == extraLocations(0).read)
            // traces that remain have same effect because x and extraLoc have same value
            // so this assert had no effect
            //    synthAssertTerminal(x == borrowedLoc.read)
            // we could fix it by setting the permutation just so; see above

            synthAssertTerminal(x == extraLocations(0).read)  // enocurage? to keep a value for 1 or 2 operations
            
            skdprint(extraStorage.mkString("push("+x.toString()+") : (", ", ", ")") + " borrowedVal:[" + borrowedVal.toString() + "]")
        }

        def pop(restore: List[A]) = {
            synthAssertTerminal(reference.size > 0)
            val refPoppedVal = reference.pop
            val refBorrowedLoc = locations.pop
            
            val potentialBorrowedLocations = new ArrayBuffer[Location[A]]
            for(location <- extraLocations) {
                potentialBorrowedLocations ++= location.read.locations
            }
            synthAssertTerminal(potentialBorrowedLocations.length > 0)
            val borrowedLoc = !!(potentialBorrowedLocations)
            synthAssertTerminal(borrowedLoc == refBorrowedLoc)

            // val borrowedVal = refBorrowedLoc.read

            var allLocations = List(refBorrowedLoc) ++ extraLocations
            val values = new ListBuffer[A]
            var found = false
            values ++= restore
            for(location <- allLocations) {
                val v = location.read
                values += v
                if(refPoppedVal == v)
                    found = true
            }
            synthAssertTerminal(found)

            skdprint(extraStorage.mkString("pop: (", ", ", ")") + " borrowdVal:[" + refBorrowedLoc.read.toString() + "] returns:" + refPoppedVal.toString())

            // this shuffle of values among allLocations will also restore the value
     
            //for(location <- allLocations) {
            //    location.write(values.remove(!!(values.length)))
            //}
            
            refBorrowedLoc.write(values.remove(popp1))//!!(values.length))) 
            extraLocations(0).write(values.remove(popp2))//!!(values.length))) 
            // extraLocations(1).write(values.remove(!!(values.length)))    one extra need not be restored
            
            skdprint(extraStorage.mkString("pop: (", ", ", ")") + " borrowdVal:[" + refBorrowedLoc.read.toString() + "] returns:" + refPoppedVal.toString())
            // determinization: attempt to go from 2 traces to 1
            // successful: this looks like a good invariant to infer!
            // at the end of the execution, it does no tmatter how we restore, hence we had two traces
            //- synthAssertTerminal(refPoppedVal == extraLocations(0).read)
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
        stack.push(origin, mkLocations(origin.children))
        
        synthAssertTerminal(origin.children(0)==root)
        
        var current  = root

        var step = 0
        while(current != origin) {
            synthAssertTerminal(step < 10)
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
                // last two arguments give node whose children fields are available to the stack as borrowed locations
                stack.push(current, mkLocations(current.children))
                skdprint("graph after push:" + g.toString())
                current = next
            } else {
                // backtrack
                skdprint("graph before pop:" + g.toString())
                current = stack.pop(List(current))
            }
        }
    }

    val test_generator = NullTestGenerator
}

object Dfs {
    def main(args: Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfsSketch())
    }
}

