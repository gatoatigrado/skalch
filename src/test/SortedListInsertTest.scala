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

    def printArray() : String = {
        var result = ""
        for (i <- 0 until length) {
            result += (if (result.isEmpty) "" else ", ") + arr(i)
        }
        result
    }

    def insert(idx : Int, v : Int) {
        skdprint("")
        skdprint("before insert:\n" + printArray())
        length += 1
        assert(idx < length, "bad index. index=" + idx + "; length=" + length)
        // NOTE - buggy sketch, see if I can detect it
        for (i <- (length - 2) to idx by -1) {
            arr(i + 1) = arr(i)
        }
        arr(idx) = v
        skdprint("after insert " + v + " at " + idx + ":\n" + printArray())
    }

    def get_insert_index_inner(v : Int) : Int = {
        var step = length / 2
        var idx = step / 2
        while (idx < length) {
            step /= 2
            var other = arr(idx)
            if (v <= other) {
                // an index at or before $other$
                return !!(idx + 1)
            } else {
                // an index after $other$
                // $idx < length$ from loop, so $untilv > 0$
                if (idx == length - 1) {
                    return length
                } else {
                    // sometimes returns zero for the solution
                    val oracle_value = `!!d`(length - idx)
                    assert(oracle_value + idx < length/* - idx + idx*/)
                    return oracle_value + idx + 1
                }
            }
        }
        idx
    }

    def get_insert_index(v : Int) : Int = {
        val idx = get_insert_index_inner(v)
        assert(idx <= length)
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
