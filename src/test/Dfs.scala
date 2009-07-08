package test
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

import scala.collection.mutable.Stack
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Buffer
import scala.runtime.RichInt

class DfsSketch() extends DynamicSketch {
    def dysketch_main() = {
        val g = new Graph(2)
        g.checkpoint

        dfs(g)

        synthAssertTerminal(g.unchanged)
        
        true
    }

    class Graph(n: Int, seed: Long = 1983) {
        val nodes = new ArrayBuffer[Node]

        val root = {
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
    }

    class Node(val name: String, newChildren: Node*) {
        var visited  = false
        var children = new ArrayBuffer[Node]()
        var checkpointChildren: Buffer[Node] = null

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
    }

    abstract class Location[A] {
        def read(): A
        def write(x: A)
    }

    class GeneralLocation[A](val reader: () => A, val writer: (A) => Unit) extends Location[A] {
        def read() = {
            reader()
        }

        def write(x: A) {
            writer(x)
        }
    }

    class BufferLocation[A](buffer: Buffer[A], index: Int) extends Location[A] {
        override def read() = {
            buffer(index)
        }

        override def write(x: A) {
            buffer(index) = x
        }
    }

    class BlackHole[A >: Null] extends Location[A] {
        override def read(): A = {
            assert(false, "You can't get anything out of a black hole!")
            null
        }

        override def write(x: A) { }
    }

    class ParasiticStack[A](val givenLocations: Seq[Location[A]]) {
        val locations = new ArrayBuffer[Location[A]]
        locations ++= givenLocations

        def push(x: A) {
            skdprint_loc("push")
            !!(locations).write(x)

            val storage = locations.clone

            for(value <- locations.map(l => l.read)) {
                storage.remove(!!(storage.length)).write(value)
            }

            /*
            for(location <- locations) {
                skdprint(location.read.name)
            }
            */
        }

        def pop() = {
            skdprint_loc("pop")
            val value = !!(locations).read

            val storage = locations.clone

            for(value <- locations.map(l => l.read)) {
                storage.remove(!!(storage.length)).write(value)
            }

            value
        }
    }

    /*
    a class where you add locations over time
    */

    class AngelicStack[A]() {
        val cheat = new ArrayBuffer[A]

        def push(x: A) {
            cheat += x
        }

        def pop() = {
            cheat.remove(!!(cheat.length))
        }
    }

    def dfs(g: Graph) {
        val root   = g.root
        val origin = new Node("origin", root)

        var current  = root

        val stack = new Stack[Node]
        stack.push(origin)

        val locations = new ArrayBuffer[Location[Node]]

        for(node <- g.nodes) {
            for(i <- 0 until node.children.length) {
                locations += new BufferLocation(node.children, i)
            }
        }

        // we may need an extra variable??!? who knows
        var v0: Node = null
        locations += new GeneralLocation[Node](() => v0, (x) => v0 = x)
        
        val mystack = new ParasiticStack[Node](locations)
        mystack.push(origin)

        var step = 0
        while(current != origin) {
            synthAssertTerminal(step < g.nodes.length * 2)
            step += 1

            for(child <- current.children) {
                synthAssertTerminal(child != null)
            }


            if(current.visited) {
                skdprint("Backtracking to: " + current.name)
            } else {
                skdprint("Visiting: " + current.name)
                current.visit()
            }

            var next: Node = null

            for(child <- current.children if !child.visited && next == null) {
                next = child
            }

            if(next == null) {
                val previous = mystack.pop
                val correct  = stack.pop

                synthAssertTerminal(previous == correct)

                current = previous
            } else {
                stack.push(current)
                mystack.push(current)
                current = next
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
