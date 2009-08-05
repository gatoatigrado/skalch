package skalch_old

import ec.util.ThreadLocalMT
import skalch.DynamicSketch

import sketch.dyn.BackendOptions
import sketch.util._

class BitonicSortSketch(val nsteps : Int, val tg_array_length : Int,
        val num_tests : Int) extends DynamicSketch
{
    val in_lengths = new InputGenerator(untilv=(1 << 30))
    val in_values = new InputGenerator(untilv=(1 << 30))

    val swap_first_idx = new HoleArray(num=nsteps, untilv=tg_array_length)
    val swap_second_idx = new HoleArray(num=nsteps, untilv=tg_array_length)
    val in_arr = new Array[Int](tg_array_length)

    def dysketch_main() = {
        val array_len : Int = in_lengths()

        // scala efficiency
        {
            var i = 0
            while (i < array_len) {
                in_arr(i) = in_values() : Int
                i += 1
            }
        }
        //val in_arr = (for (i <- 0 until array_len) yield in_values()).toArray

        // print("input array", in_arr.toString)
        {
            var a = 0
            while (a < nsteps) {
//         for (a <- 0 until nsteps) {
                // NOTE / ntung - if the hole array is null, then it will just
                // throw an exception, which is caught. This is probably not
                // desirable. The compiler should track which indices are
                // synthesized with holes, and then only allow those to
                // throw exceptions
                val first_swap : Int = swap_first_idx(a)
                val second_swap : Int = swap_second_idx(a)
                // print("step " + a + ", swap (" + first_swap + ", " + second_swap + ")")
                if (in_arr(second_swap) < in_arr(first_swap)) {
                    val tmp = in_arr(first_swap)
                    in_arr(first_swap) = in_arr(second_swap)
                    in_arr(second_swap) = tmp
                }
                a += 1
            }
            def format_swap_indices(hole_arr : HoleArray) = {
                val arr = (for (i <- 0 until nsteps) yield hole_arr(i)).toArray
                ("" /: arr)((x : String, y : Int) =>
                    if (x.isEmpty) y.toString else x + ", " + y)
            }
            skdprint_loc("arrays")
            skdprint(format_swap_indices(swap_first_idx))
            skdprint(format_swap_indices(swap_second_idx))
        }
        // print("sorted array", in_arr.toString)
        var a = 0
        var num_violations = 0
        while (a < array_len - 1) {
            num_violations += (if (in_arr(a) <= in_arr(a + 1)) 0 else 1)
            a += 1
        }
        skAddCost(num_violations)
//         for (a <- 0 until (array_len - 1))
//              synthAssertTerminal(in_arr(a) <= in_arr(a + 1))
        (num_violations == 0)
    }

    val test_generator = new TestGenerator {
        // this is supposed to be expressive only,
        // recover it with Java reflection if necessary
        def set(x : Int) {
            put_input(in_lengths, x)
            for (i <- 0 until x) put_input(in_values,
                BitonicSortTest.mt.get().nextInt() >> 20)
        }
        def tests() {
            for (i <- 0 until num_tests)
                test_case(tg_array_length : java.lang.Integer)
        }
    }
}

object BitonicSortTest {
    val mt = new ThreadLocalMT()

    object TestOptions extends cli.CliOptionGroup {
        add("--array_length", "length of array")
        add("--num_steps", "number of steps allowed")
        add("--num_tests", 100 : java.lang.Integer,
            "number of randomly generated input sequences to test")
    }

    def main(args : Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new BitonicSortSketch(
            opts.long_("num_steps").intValue,
            opts.long_("array_length").intValue,
            opts.long_("num_tests").intValue))
    }
}
