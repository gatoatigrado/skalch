package test

import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util._

class Node(val value : Int) {
    var next: Node = null
    override def toString = "Node[" + value + "]"
    def listString(visited : List[Node]): String = {
        if (visited.exists(n => n.value == value))
            "already visited " + value
        else if (next == null) value.toString
        else value + " -> " + next.listString(this :: visited)
    }
}

class RevList1(val length : Int) extends DynamicSketch {
    def checkReversed(len: Int, rev: Node): Unit = {
        var curNum = len
        var curNode = rev
        while (curNode != null) {
            synthAssertTerminal(curNode.value == curNum)
            curNum -= 1
            curNode = curNode.next
        }
        synthAssertTerminal(curNum == 0)
    }

    def buildList(size: Int): (Node, Array[Node]) = {
        val nodes = new Array[Node](size + 1)
        var head: Node = null
        var last: Node = null
        var i = 1
        while (i <= size) {
            var cur = new Node(i)
            nodes(i - 1) = cur
            if (head == null)
                head = cur
            if (last != null)
                last.next = cur
            last = cur
            i += 1
        }
        (head, nodes)
    }

    def listLength(l: Node): Int = {
        if (l == null) 0
        else listLength(l.next) + 1
    }

    def reverse(l : Node, all_nodes : Array[Node]) : Node = {
        val len = listLength(l)
        var i = 0
        try {
            while (i < len) {
                // two different styles
                !!(all_nodes).next =
                    all_nodes(!!(all_nodes.length))
                i += 1
            }
        } catch {
            case ex : java.lang.NullPointerException =>
                synthAssertTerminal(false)
        }
        return !!(all_nodes)
    }

    def dysketch_main() = {
        val (list, nodes) = buildList(length)
        val reversed = reverse(list, nodes)
        checkReversed(length, reversed)
        skprint("reversed", reversed.listString(List()))
        true
    }

    val test_generator = NullTestGenerator()
}

object RevListTest {
    object TestOptions extends CliOptGroup {
        add("--list_length", "length of list")
    }

    def main(args : Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new RevList1(opts.long_("list_length").intValue))
    }
}
