package test

import ec.util.ThreadLocalMT
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class BitonicSort(val nsteps : Int, val tg_array_length : Int,
        val num_tests : Int) extends DynamicSketch
{
    val in_lengths = new InputGenerator(untilv=(1 << 30))
    val in_values = new InputGenerator(untilv=(1 << 30))

    val swap_first_idx = hole_array(num=nsteps, untilv=tg_array_length)
    val swap_second_idx = hole_array(num=nsteps, untilv=tg_array_length)

    def dysketch_main() = {
        val array_len : Int = in_lengths()
        val in = (for (i <- 0 until array_len) yield in_values()).toArray
        // print("input array", in.toString)
        for (a <- 0 until nsteps) {
            // NOTE / ntung - if the hole array is null, then it will just throw an exception,
            // which is caught. This is probably not desirable. The compiler should track
            // which indices are synthesized with holes, and then only allow those to throw exceptions
            val first_swap : Int = swap_first_idx(a)()
            val second_swap : Int = swap_second_idx(a)()
            // print("step " + a + ", swap (" + first_swap + ", " + second_swap + ")")
            if (in(second_swap) < in(first_swap)) {
                val tmp = in(first_swap)
                in(first_swap) = in(second_swap)
                in(second_swap) = tmp
            }
        }
        // print("sorted array", in.toString)
        for (a <- 0 until (array_len - 1)) synthAssertTerminal(in(a) <= in(a + 1))
        true
    }

    override def solution_str() = {
        "swap first idx " + (for (h <- swap_first_idx) yield h()).toArray + ", " +
        "swap second idx " + (for (h <- swap_second_idx) yield h()).toArray
    }

    val test_generator = new TestGenerator {
        val mt = new ThreadLocalMT()
        // this is supposed to be expressive only, recover it with Java reflection if necessary
        def set(x : Int) {
            put_input(in_lengths, x)
            for (i <- 0 until x) put_input(in_values, mt.get().nextInt() >> 20)
        }
        def tests() {
            for (i <- 0 until num_tests) test_case(tg_array_length : java.lang.Integer)
        }
    }
}

object BitonicSortTest {
    object TestOptions extends CliOptGroup {
        add("--array_length", "length of array")
        add("--num_steps", "number of steps allowed")
        add("--num_tests", 1 : java.lang.Integer, "number of randomly generated input sequences to test")
    }

    def main(args : Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val opts = TestOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new BitonicSort(opts.int_("num_steps"),
                opts.int_("array_length"), opts.int_("num_tests")))
    }
}
