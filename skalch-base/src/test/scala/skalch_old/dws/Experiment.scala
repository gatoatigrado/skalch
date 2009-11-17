/*
package skalch_old.dws
import skalch.DynamicSketch
import scala.collection.mutable._

import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

/*
create
insert
accessmin
deletemin
decreasekey
delete
merge
*/

class Necklace[A](elements: A*) extends Seq[A] {
    var clasp: Pearl[A] = null

    for(element <- elements) {
        append(element)
    }

    def append(element: A) {
        if(clasp == null) {
            clasp = new Pearl[A](element)
        } else {
            clasp.append(element)
        }
    }

    def iterator() = {
        new NecklaceIterator()
    }

    def apply(idx: Int) = {
        if(clasp == null) {
            throw new java.util.NoSuchElementException()
        } else {
            clasp(idx)
        }
    }

    def length() = {
        size
    }

    class NecklaceIterator() extends Iterator[A] {
        var current = clasp

        def hasNext() = {
            current != null
        }

        def next() = {
            val value = current.elem

            if(current.next == clasp) {
                current = null
            } else {
                current = current.next
            }

            value
        }
    }


    class Pearl[A](element: A, previous: Pearl[A]) {
        def this(element: A) = this(element, null)

        val elem: A        = element
        var prev: Pearl[A] = previous
        var next: Pearl[A] = null

        if(prev == null) {
            prev = this
            next = this
        } else {
            next = prev.next
            prev.next = this

            next.prev = this
        }

        assert(localCheck())

        def cutForwards(): (Pearl[A], Pearl[A]) = {
            val a = this
            val b = next

            this.next = null
            next.prev = null

            (a, b)
        }

        def cutBackwards(): (Pearl[A], Pearl[A]) = {
            val a = prev
            val b = this

            this.prev = null
            prev.next = null

            (a, b)
        }

        def attachForward(newNext: Pearl[A]) {
            assert(this.next == null)
            assert(newNext.prev == null)

            this.next = newNext
            newNext.prev = this
        }

        def splice(other: Pearl[A]) {

        }

        def apply(idx: Int): A = {
            if(idx == 0) {
                elem
            } else if(next == null) {
                throw new java.util.NoSuchElementException()
            } else {
                next(idx - 1)
            }
        }

        def localCheck() = {
            prev.next == this && next.prev == this
        }

        def insertAfter(newElem: A) = {
            new Pearl[A](newElem, this)
        }

        def insertBefore(newElem: A) = {
            this.prev.insertAfter(newElem)
        }

        def append(newElem: A) {
            insertBefore(newElem)
        }

        def foreach(f: A => Unit) {
            var current = this
            do {
                f(current.elem)
                current = current.next
            } while(current != this)
        }
    }
}

/*
class ExplodingSet[A] extends HashSet[A] {
    override def +=(elem: A) {
        if(contains(elem)) {
            throw new java.lang.Exception("Boom!!!!")
        } else {
            super += elem
        }
    }
}

object Tree {
    def isWellFormed(tree: Tree[_]) {
        val set = new ExplodingSet[Tree[_]]
    }
}

class Tree[A](var parent: Tree[A], var children: Necklace[Tree[A]]) {
    def this() = this(null, null)

    if(children == null) {
        children = new Necklace[Tree[A]]
    }

    if(parent != null) {
        parent.children.append(this)
    }

    def siblings = parent.children
}

class FibonacciHeap {
    val storage = new Necklace[Tree[Int]]

    def isWellFormed = {
        for(tree <- storage) {
            if(!tree.isWellFormed()) return false
        }

        true
    }

    def insert(value: A) {
    }

    def someKey() = {
        cheat.keys(oracle(cheat.keys.size - 1))
    }

    def someValue() = {
        cheat(someKey())
    }

    def accessmin() = {
        someValue()
    }

    def deletemin() = {
        cheat.removeEntry(someKey())
    }

    def decreasekey(item: A, newKey: Int) {
        //cheat.removeEntry()
    }

    def merge {
    }
}
*/

class ExperimentalSketch() extends DynamicSketch {
    val skassert = synthAssert _

    def dysketch_main() = {
        val n = new Necklace[Int](1)
        n.append(2)
        n.append(3)
        n.append(4)

        Console.println(n(2))

        assert(n.clasp.elem == 1)
        assert(n.clasp.next.elem == 2)
        assert(n.clasp.next.next.elem == 3)
        assert(n.clasp.next.next.next.elem == 4)
        assert(n.clasp.next.next.next.next.elem == 1)
        assert(n.clasp.prev.elem == 4)
        assert(n.clasp.prev.prev.elem == 3)
        assert(n.clasp.prev.prev.prev.elem == 2)
        assert(n.clasp.prev.prev.prev.prev.elem == 1)

        true
    }

    val test_generator = NullTestGenerator
}

object Experiment {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new ExperimentalSketch())
    }
}
*/
