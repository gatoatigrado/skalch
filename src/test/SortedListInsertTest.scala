/** sad as it is, I still have problems writing this
 * @author gatoatigrado (nicholas tung) [ntung at ntung]
 */
package test

import ec.util.ThreadLocalMT
import sketch.dyn.BackendOptions
import sketch.util._

class SortedListInsertSketch(val list_length : Int,
    val num_tests : Int) extends RBTreeSketchBase
{
    val arr = new Array[Int](list_length)
    var length = 0

    def insert(idx : Int, v : Int) {
        length += 1
        // NOTE - buggy sketch, see if I can detect it
        for (i <- idx until (length - 1)) {
            arr(i + 1) = arr(i)
        }
        arr(idx) = v
    }

    def dysketch_main() : Boolean = {
        length = 0
        for (a <- 0 until list_length) {
            val in = next_int_input()
            insert(!!(length + 1), in)
        }

        // check is sorted
        skdprint("unsorted list: " + arr.toString)
        for (i <- 0 until (length - 1)) {
            if (arr(i) > arr(i + 1)) { return false }
        }
        skdprint("sorted list: " + arr.toString)
        true
    }

    val test_generator = new TestGenerator() {
        def set() {
            for (a <- 0 until list_length) {
                val v = SortedListInsertTest.mt.get().nextInt(100)
                put_default_input(v)
            }
        }
        def tests() { for (i <- 0 until num_tests) test_case() }
    }
}

object SortedListInsertTest {
    val mt = new ThreadLocalMT()

    object TestOptions extends CliOptGroup {
        import java.lang.Integer
        add("--list_length", 10 : Integer, "length of list")
        add("--num_tests", 1 : Integer, "number of tests")
    }

    def main(args : Array[String]) = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new SortedListInsertSketch(
            opts.long_("list_length").intValue,
            opts.long_("num_tests").intValue))
    }
}
