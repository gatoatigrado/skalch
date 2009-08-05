/** sad as it is, I still have problems writing this
 * @author gatoatigrado (nicholas tung) [ntung at ntung]
 */
package skalch_old

import skalch.DynamicSketch

import ec.util.ThreadLocalMT
import sketch.dyn.BackendOptions
import sketch.util._

class SortedListInsertSketch(val list_length : Int,
    val num_tests : Int) extends DynamicSketch
{
    val arr = new Array[Int](list_length)
    var length = 0

    def printArray() : String = {
        var result = ""
        for (i <- 0 until length) {
            result += (if (result.isEmpty) "" else ", ") + arr(i)
        }
        result
    }

    def insert(idx : Int, v : Int) {
        length += 1
        assert(idx < length, "bad index. index=" + idx + "; length=" + length)
        // NOTE - buggy sketch, see if I can detect it
        for (i <- (length - 2) to idx by -1) {
            arr(i + 1) = arr(i)
        }
        arr(idx) = v
        skdprint("after insert array: " + printArray())
    }

    def get_insert_index(v : Int) : Int = {
        var step = length / 2
        var idx = step
        while (idx < length) {
            step /= 2
            val other = arr(idx)
            if (v < other) {
                // an index at or before $other$
                if (idx == 0) {
                    skdprint("at zero, return zero")
                    return 0
                } else {
                    val other2 = arr(idx - 1)
                    if (other2 <= v) {
                        skdprint("returning idx " + idx)
                        return idx
                    } else {
                        // search left subtree
                        skdprint("search left subtree")
                        idx -= Math.max(1, step)
                    }
                    //return !!(idx + 1)
                }
            } else {
                // an index after $other$
                // $idx < length$ from loop, so $untilv > 0$
                if (idx == length - 1) {
                    skdprint("at end, return length")
                    return length
                } else {
                    // search right subtree
                    skdprint("search right subtree")
                    idx += Math.max(1, step)
                }
            }
        }
        idx
    }

    def dysketch_main() : Boolean = {
        length = 0
        var sum = 0
        for (a <- 0 until list_length) {
            val in = next_int_input()
            sum += in
            insert(get_insert_index(in), in)
        }

        // check is sorted
        var sorted_sum = arr(length - 1)
        for (i <- 0 until (length - 1)) {
            sorted_sum += arr(i)
            if (arr(i) > arr(i + 1)) { return false }
        }
        skAddCost(Math.abs(sorted_sum - sum))
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

    object TestOptions extends cli.CliOptionGroup {
        import java.lang.Integer
        add("--list_length", 10 : Integer, "length of list")
        add("--num_tests", 10 : Integer, "number of tests")
    }

    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new SortedListInsertSketch(
            opts.long_("list_length").intValue,
            opts.long_("num_tests").intValue))
    }
}
