package test.skalch_old.bench

import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

import java.lang.Integer

/*
This was kinda in a hurry, but as of rev 30d2c2117dc84fb323c84b16347eed8b57d0c549+,
output looks good.

    [skdprint] input: 4
    [skdprint] input: 2
    [skdprint] insert left child of 0
    [skdprint] input: 6
    [skdprint] insert right child of 0
    [skdprint] input: 5
    [skdprint] insert left child of 2
    [skdprint] input: 8
    [skdprint] insert right child of 2
    [skdprint] input: 3
    [skdprint] insert right child of 1
    [skdprint] input: 7
    [skdprint] insert left child of 4
    [skdprint] input: 1
    [skdprint] insert left child of 1
*/

class BintreeInsertSketch extends DynamicSketch {
    class BintreeNode(var value : Int) {
        var leftChild, rightChild : BintreeNode = null
    }

    var nnodes = 0
    val node_arr = (for (i <- 0 until 10) yield new BintreeNode(-1)).toArray
    val input_length = BintreeOptions("num_nodes")

    def new_node(value : Int) : BintreeNode = {
        val next : BintreeNode = node_arr(nnodes)
        next.value = value
        next.leftChild = null
        next.rightChild = null
        nnodes += 1
        next
    }

    def find(node__ : BintreeNode, tofind : Int) : Boolean = {
        var node = node__
        while (node != null) {
            if (node.value == tofind) {
                return true
            } else if (tofind < node.value) {
                node = node.leftChild
            } else {
                node = node.rightChild
            }
        }
        false
    }

    def allPresent(node : BintreeNode) : Boolean = {
        for (i <- 1 to input_length) {
            if (!find(node, i)) {
                return false
            }
        }
        true
    }

    def dysketch_main() : Boolean = {
        nnodes = 0
        var root : BintreeNode = null
        for (i <- 0 until input_length) {
            val value = next_int_input()
            // skdprint("input: " + value)
            val node = new_node(value)
            if (root == null) {
                root = node
            } else {
                val insert_idx = !!(nnodes)
                val to_insert = node_arr(insert_idx)
                if (!!()) {
                    // skdprint("insert left child of " + insert_idx)
                    to_insert.leftChild = node
                } else {
                    // skdprint("insert right child of " + insert_idx)
                    to_insert.rightChild = node
                }
                /*
                NOTE / ntung - this is a really weird situation in which
                Scala does something weird for skdprint(to_insert.toString),
                which gives me some assertion failure
                val str_result = to_insert.toString
                skdprint(str_result)
                */
            }
        }
        allPresent(root)
    }

    val test_generator = new TestGenerator() {
        import ec.util.ThreadLocalMT.mt
        def set() {
            val arr = new Array[Int](input_length)
            def swap(i : Int, j : Int) {
                val tmp = arr(i)
                arr(i) = arr(j)
                arr(j) = tmp
            }
            for (i <- 0 until input_length) {
                arr(i) = i + 1
            }
            for (i <- 0 until input_length) {
                swap(mt().nextInt(input_length), mt().nextInt(input_length))
            }
            for (i <- 0 until input_length) {
                print("test input", arr(i) : Integer)
                put_default_input(arr(i))
            }
        }
        def tests() { test_case() }
    }
}

object BintreeInsertTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        BintreeOptions.result = BintreeOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new BintreeInsertSketch())
    }
}

object BintreeOptions extends cli.CliOptionGroup {
    var result : cli.CliOptionResult = null
    import java.lang.Integer
    add("--num_nodes", 8 : Integer, "number of nodes to insert")
    def apply(x : String) : Int = result.long_(x).intValue
}
