package test.df

import skalch.DynamicSketch
import sketch.util._

abstract class AbstractDfSketch() extends DynamicSketch {
    val example_idx = AbstractDfOptions.result.long_("example_idx").intValue
    val examples = Array(
        Array(White(), Blue(), Red()),
        Array(Blue(), White(), Red(), Blue(), Red(), White(), Red()),
        Array(Red(), White(), Blue(), Red()),
        Array[Pebble]()
        )
    val num_buckets = (if (example_idx == -1) {
        AbstractDfOptions.result.long_("num_buckets").intValue
    } else {
        examples(example_idx).length
    })

    case class Pebble(val order : Int, val color : String) {
        override def toString = color
    }

    case class Red() extends Pebble(0, "red")
    case class White() extends Pebble(1, "white")
    case class Blue() extends Pebble(2, "blue")

    val buckets = new Array[Pebble](num_buckets)

    def read_from_input() = {
        var i = 0
        while (i < num_buckets) {
            buckets(i) = next_int_input() match {
                case 0 => Red()
                case 1 => White()
                case 2 => Blue()
            }
            i += 1
        }
    }

    def isCorrect(n : Int) : Boolean = {
        // (buckets.slice(0, n - 1) zip buckets.slice(1, n)).forall(x => x._1 <= x._2)
        var i = 0
        while (i < (n - 1)) {
            if (buckets(i).order > buckets(i + 1).order) {
                return false
            }
            i += 1
        }
        true
    }

    def isCorrect() : Boolean = isCorrect(num_buckets)

    def swap(i : Int, j : Int) {
        val tmp = buckets(i)
        buckets(i) = buckets(j)
        buckets(j) = tmp
    }

    def useful_swap(i : Int, j : Int) {
        synthAssertTerminal(i != j)
        swap(i, j)
    }

    def swap_ordered(i : Int, j : Int) {
        synthAssertTerminal(i < j)
        swap(i, j)
    }

    final def dysketch_main() : Boolean = {
        read_from_input()
        df_main()
        isCorrect(buckets.length)
    }

    def df_main()

    val test_generator = new TestGenerator() {
        def set() {
            val example_arr = examples(example_idx)
            for (a <- 0 until num_buckets) {
                put_default_input(example_arr(a).order)
            }
        }
        def tests() { test_case() }
    }
}

object AbstractDfOptions extends cli.CliOptionGroup("df", "Dutch flag options") {
    var result : cli.CliOptionResult = null
    import java.lang.Integer
    add("--num_buckets", 10 : Integer, "number of buckets (for random generation)")
    add("--example_idx", -1 : Integer, "select an example, overriding the num_buckets")
}
