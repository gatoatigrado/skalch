/*
// @code standards ignore file
package skalch_old.dws

import skalch.DynamicSketch
import scala.collection.mutable._

import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

/** @author Casey Rodarmor */
class NextPermutationSketch() extends DynamicSketch {
    val skassert = synthAssertTerminal _

    def dysketch_main() = {
        val l  = 5
        val ps = permutations(l).toList.sort(lt)

        for(i <- 0 until (ps.length - 1)) {
            val my_next  = next(ps(i))
            val the_next = ps(i + 1).toArray
            for(i <- 0 until l) {
                skassert(my_next(i) == the_next(i))
            }
        }

        true
    }

    def lt(a: Array[Int], b: Array[Int]): Boolean = {
        for(i <- 0 until a.length) {
            if(a(i) < b(i)) {
                return true
            }

            if(a(i) > b(i)) {
                return false
            }
        }

        return false
    }

    def fac(n: Int): Int = {
        if(n == 0) {
            1
        } else {
            n * fac(n - 1)
        }
    }

    def join(a: Array[_]) = {
        var s = ""
        for(o <- a) s += o
        s
    }

    // returns an unordered list of permutations
    def permutations(length: Int) = {
        val sequence = Array.range(0, length)

        for(i <- 0 until fac(sequence.length)) yield permutation(i, sequence)
    }

    // returns the nth permutation of a
    def permutation(n: Int, a: Array[Int]) = {
        var k = n
        val p = a.toArray

        for(j <- 2 to a.length) {
            var temp = p(j - 1)
            p(j - 1) = p(k % j)
            p(k % j) = temp
            k = k / j
        }

        p
    }

    // finds the lexically next permutation of p
    def next(p: Array[Int]) = {
        var swap_index = !!(p.length - 1)

        skdprint(join(p) + " split at " + swap_index)

        /* // don't tell anybody, but this is the answer:
        var swap_index = 0
        for(i <- 0 until p.length - 1) {
            if(p(i) < p(i + 1)) {
                swap_index = i
            }
        }
        */

//         val unchanged = p.subArray(0, swap_index)
        // FIXME - bug in 2.8.0 latest
        val unchanged = new Array[Int](swap_index)
        for (i <- 0 until swap_index) {
            unchanged(i) = p(i)
        }

        val candidates = for(entry @ (value, index) <- p.zipWithIndex.slice(swap_index + 1, p.length) if value > p(swap_index)) yield entry

        skassert(candidates.length > 0)

        val replacement       = candidates.min._1
        val replacement_index = candidates.min._2

        val rest = p.slice(swap_index, replacement_index) ++ p.slice(replacement_index + 1, p.length)

        unchanged ++ Array(replacement) ++ scala.util.Sorting.stableSort(rest.asInstanceOf[Seq[Int]])
    }

    val test_generator = NullTestGenerator
}

object NextPermutation {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new NextPermutationSketch())
    }
}
*/
