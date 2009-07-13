package test.df

import skalch.DynamicSketch

abstract class AbstractDfSketch(val num_buckets : Int) extends DynamicSketch {
    case class Pebble(val order : Boolean, val color : String) {
        override def toString = color
    }

    case class Red() extends Pebble(0, "red")
    case class White() extends Pebble(1, "white")
    case class Blue() extends Pebble(2, "blue")

    val buckets = new Array[Pebble](num_buckets)

    def read_from_input() = {
        var i = 0
        while (i < num_buckets) {
            buckets(i) = next_int_input()
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
        }
        true
    }

    val tests = new TestGenerator() {
        val arr = Array(White(), Blue(), Red())
        def set() {
            for (a <- 0 until num_buckets) {
                put_default_input(arr(a))
            }
        }
        def tests() { test_case() }
    }
}
