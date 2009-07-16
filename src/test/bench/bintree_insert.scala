package test.bench

import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

import java.lang.Integer

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

    def dysketch_main() : Boolean = {
        nnodes = 0
        var root : BintreeNode = null
        for (i <- 0 until input_length) {
            val value = next_int_input()
            DebugOut.print_mt("next int input", value : Integer)
            val node = new_node(value)
            if (root == null) {
                root = node
            } else {
                val to_insert = node_arr(!!(nnodes))
                /*
                NOTE / ntung - this is a really weird situation in which
                Scala does something weird for skdprint(to_insert.toString),
                which gives me some assertion failure
                val str_result = to_insert.toString
                skdprint(str_result)
                */
            }
        }
        //allPresent(root)
        true
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
