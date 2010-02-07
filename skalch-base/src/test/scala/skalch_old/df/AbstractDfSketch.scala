package skalch_old.df

import skalch.DynamicSketch
import sketch.util._

abstract class AbstractDfSketch() extends DynamicSketch {
    var num_buckets = -1
    var max_num_buckets = Math.max(30,
        AbstractDfOptions.result.long_("num_buckets").intValue)
    val num_buckets_input = new InputGenerator(max_num_buckets)

    case class Pebble(val order : Int, val color : String) {
        override def toString = color
    }

    case class Red() extends Pebble(0, "red")
    case class White() extends Pebble(1, "white")
    case class Blue() extends Pebble(2, "blue")

    val buckets = new Array[Pebble](max_num_buckets)

    def abbrev_str() : String =
        (("" /: (buckets.view(0, num_buckets)))(_ + ", " + _)).substring(2)

    def read_from_input() = {
        var i = 0
        num_buckets = num_buckets_input()
        while (i < num_buckets) {
            buckets(i) = next_int_input() match {
                case 0 => Red()
                case 1 => White()
                case 2 => Blue()
                case v => assert(false, "invalid input " + v); null
            }
            i += 1
        }
    }

    def isCorrect(n : Int) : Boolean = {
        // (buckets.slice(0, n - 1) zip buckets.slice(1, n)).forall(x => x._1 <= x._2)
        var i = 0
        while (i < (n - 1)) {
            assert((i + 1) < num_buckets)
            assert(buckets(i) != null)
            assert(buckets(i + 1) != null)
            if (buckets(i).order > buckets(i + 1).order) {
                return false
            }
            i += 1
        }
        true
    }

    def isCorrect() : Boolean = isCorrect(num_buckets)

    def swap(i : Int, j : Int) {
        assert((i < num_buckets) && (j < num_buckets))
        //skdprint("swap(buckets[" + i + "]=" + buckets(i) +
        //    " with buckets[" + j + "]=" + buckets(j) + " )")
        assert((buckets(i) != null) && (buckets(j) != null))
        val tmp = buckets(i)
        buckets(i) = buckets(j)
        buckets(j) = tmp
    }

    def swap_useful(i : Int, j : Int) {
        synthAssert(i != j)
        swap(i, j)
    }

    def swap_ordered(i : Int, j : Int) {
        synthAssert(i < j)
        swap(i, j)
    }

    final def dysketch_main() : Boolean = {
        var i = 0
        df_init()
        while (i < num_total_tests) {
            read_from_input()
            df_main()
            if (!isCorrect()) {
                return false
            }
            i += 1
        }
        true
    }

    def df_init() { }
    def df_main()

    val test_generator = new TestGenerator() {
        import ec.util.ThreadLocalMT.mt
        val examples = Array(
            Array(White(), Blue(), Red()),
            Array(Blue(), White(), Red(), Blue(), Red(), White(), Red()),
            Array(Red(), White(), Blue(), Red()),
            Array[Pebble]()
            )
        val example_idx = AbstractDfOptions("example_idx")
        val rand_num_buckets = AbstractDfOptions("num_buckets")
        val num_tests = AbstractDfOptions("num_random_tests")
        val num_total_tests = num_tests + (if (example_idx != -1) 1 else 0)
        def set() {
            if (example_idx != -1) {
                val example_arr = examples(example_idx)
                put_input(num_buckets_input, example_arr.length)
                for (a <- 0 until example_arr.length) {
                    put_default_input(example_arr(a).order)
                }
            }
            for (tcidx <- 0 until num_tests) {
                put_input(num_buckets_input, rand_num_buckets)
                for (a <- 0 until rand_num_buckets) {
                    put_default_input(mt().nextInt(3))
                }
            }
        }
        def tests() { test_case() }
    }
    val num_total_tests = test_generator.num_total_tests
}

object AbstractDfOptions extends cli.CliOptionGroup("df", "Dutch flag options") {
    var result : cli.CliOptionResult = null
    import java.lang.Integer
    addOption("num_buckets", 8 : Integer, "number of buckets for random tests")
    addOption("num_random_tests", 10 : Integer, "number of random tests")
    addOption("example_idx", -1 : Integer, "select an example, overriding the num_buckets")
    def apply(x : String) : Int = result.long_(x).intValue
}
