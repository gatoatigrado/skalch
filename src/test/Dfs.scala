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

        //val e = new Node("e")
        //val d = new Node("d")
        val c = new Node("c")
        val b = new Node("b")
        val a = new Node("a", b, c)

        /*
            e.children += e
            e.children += e
            e.children += e
        */

        val root = a

        nodes ++= List(a,b,c)
        
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
                string += "]\n"
            }
            
            string
        }
    }

    class Node(val name: String, newChildren: Node*) {
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

    class AngelicStack[A]() {
        val cheat = new ArrayBuffer[A]

        def push(x: A) {
            cheat += x
        }

        def pop() = {
            cheat.remove(!!(cheat.length))
        }
    }

    class KeyholeStack[A >: Null](allowedExtraStorage: Int = 0) {
        val reference = new Stack[A]

        val extraStorage = new ArrayBuffer[A]

        var i = 0
        while(i < allowedExtraStorage) {
            extraStorage += null
            i += 1
        }
        
        val extraLocations = mkLocations(extraStorage)

        def push(x: A, to: Seq[Location[A]]) {
            reference.push(x)

            val storage = to ++ extraLocations
            !!(storage).write(x)

            val values = new ListBuffer[A]

            for(location <- storage) {
                values += location.read
            }

            for(location <- storage) {
                location.write(values.remove(!!(values.length)))
            }
        }

        def pop(from: Seq[Location[A]]) = {
            synthAssertTerminal(reference.size > 0)

            val need = reference.pop

            var found = false

            val storage = from ++ extraLocations

            for(location <- storage) {
                if(need == location.read)
                    found = true
            }

            synthAssertTerminal(found)

            val values = new ListBuffer[A]

            for(location <- storage) {
                values += location.read
            }

            for(location <- storage) {
                location.write(values.remove(!!(values.length)))
            }

            need
        }
    }

    def mkLocations[A](b: Buffer[A]): Seq[Location[A]] = (0 until b.length).map(i => new BufferLocation(b, i))

    def dfs(g: Graph) {
        val root   = g.root
        val origin = new Node("origin", root)

        var current  = root

        var extraLocations = List[Location[Node]](new GeneralLocation[Node](() => current, (x) => current = x))

        val stack = new KeyholeStack[Node](1)
        stack.push(origin, new ArrayBuffer[Location[Node]])

        var step = 0
        while(current != origin) {
            synthAssertTerminal(step < 10)
            step += 1

            skdprint(stack.extraStorage.mkString("(", ", ", ")") + "\n" + g.toString())

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
                stack.push(current, mkLocations(current.children))
                current = next
            } else {
                // backtrack
                current = stack.pop(mkLocations(current.children) ++ extraLocations)
            }
        }

        skdprint(stack.extraStorage.mkString("(", ", ", ")") + "\n" + g.toString())
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
