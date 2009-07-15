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

    class Graph(n: Int, seed: Long = 1983) {
        val nodes = new ArrayBuffer[Node]

        val d = new Node("d")
        val c = new Node("c", d)
        val b = new Node("b", d)
        val a = new Node("a", b, c)

        nodes ++= List(a,b,c,d)

        val root = a
        
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

        var uninspected = 0

        def visit() {
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

    /*
    class KeyholeStackA[A >: Null](allowedExtraStorage: Int = 0) {
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

            var storage = List() ++ extraLocations

            if(to.length > 0) {
                storage ++= List(!!(to))
            }

            !!(storage).write(x)

            val values = new ListBuffer[A]

            for(location <- storage) {
                values += location.read
            }

            for(location <- storage) {
                location.write(values.remove(!!(values.length)))
            }

            for(location <- to) {
                synthAssertTerminal(location.read != null)
            }
        }

        def pop(from: Seq[Location[A]]) = {
            synthAssertTerminal(reference.size > 0)

            val need = reference.pop

            var found = false

            for(location <- extraLocations) {
                if(need == location.read) {
                    found = true
                }
            }

            synthAssertTerminal(found)

            val storage = extraLocations ++ from

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
    */

    def mkLocations[A](b: Buffer[A]): Seq[Location[A]] = {
        val locations = new ArrayBuffer[Location[A]]

        var i = 0
        while(i < b.length) {
            locations += new BufferLocation(b, i)
            i += 1
        }

        locations
    }

    /*
    def printState(s: String, stack: KeyholeStack[Node], g: Graph) {
        skdprint(s + "\n" + stack.extraStorage.mkString("(", ", ", ")") + "\n" + g.toString())
    }
    */

    trait LocationHaver[A] {
        def locations(): Seq[Location[A]]
    }

    /*
    class KeyholeStack[A >: LocationHaver[A]]() {
        val cheat = new Stack[A]

        def push(x: Location[A], to: Seq[Location[A]]) {
            cheat.push(x.read)
        }

        def pop(from: Seq[Location[A]]) = {
            cheat.pop()
        }
    }
    */

    def dfs(g: Graph) {
        val root   = g.root
        val origin = new Node("origin", root)

        //val stack = new KeyholeStack[Node](1)
        //stack.push(origin, List[Location[Node]]())

        var current  = root
        var previous = origin

        val locations = List(new GeneralLocation(() => current,  (x: Node) => current  = x),
                             new GeneralLocation(() => previous, (x: Node) => previous = x))

        var stepsTaken = 0
        while(current != origin) {
            synthAssertTerminal(stepsTaken < 100)
            stepsTaken += 1

            if(current.visited) {
                skdprint("Backtracking to: " + current.name)
            } else {
                skdprint("Visiting: " + current.name)
                current.visit()
            }

            var next: Location[Node] = null

            while(next == null && current.uninspected < current.children.length) {
                if(!current.children(current.uninspected).visited) {
                    next = new BufferLocation(current.children, current.uninspected)
                } else {
                    current.uninspected += 1
                }
            }

            if(next != null) {
                val temp = next.read
                next.write(previous)
                previous = current
                current = temp

                // push current onto the stack:
                // /-> current -> previous -> next --\
                // \---------------------------------/
                
            } else {
                next = new BufferLocation(previous.children, previous.uninspected)

                val temp = next.read

                next.write(current)
                current = previous
                previous = temp

                // pop previous from the stack:
                // /-- previous <- next <- current <-\
                // \---------------------------------/

                current.uninspected += 1
            }
        }
    }

    def permute[A](locations: Seq[Location[A]]) {
        val values = new ArrayBuffer[A]

        for(location <- locations) {
            values += location.read
        }

        for(location <- locations) {
            location.write(values.remove(!!(values.length)))
        }
    }

    val test_generator = NullTestGenerator;
}

object Dfs {
    def main(args: Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfsSketch())
    }
}
