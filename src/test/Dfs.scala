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
        val g = new Graph(10)
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
            while(i < n * 3) {
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

    class AngelicStack[A] {
        val storage = new ArrayBuffer[A]

        def push(x: A) {
            storage += x
        }

        def pop() = {
            storage.remove(!!(storage.length))
        }
    }

    def dfs(g: Graph) {
        val root   = g.root
        val origin = new Node("origin", root)

        var current  = root

        val stack = new Stack[Node]
        stack.push(origin)

        val mystack = new AngelicStack[Node]
        mystack.push(origin)

        while(current != origin) {
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
