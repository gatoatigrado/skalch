package angelic.conceptual

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class StrassenMultiplySketch(val first_nadds : Int, val nmultiplies : Int,
        val final_nadds : Int) extends AngelicSketch
{
    class Matrix (val base : Array2D[Int]) {
        def chop(nx : Int, ny : Int)  : Array[Matrix] = {
            val result = new Array[Matrix](nx * ny)
            for (x <- 0 until nx) {
                for (y <- 0 until ny) {
                    val xfrom = (x * base.width) / nx
                    val xuntil = ((x + 1) * base.width) / nx
                    val yfrom = (y * base.width) / ny
                    val yuntil = ((y + 1) * base.width) / ny
                    val matrix = new Matrix(xuntil - xfrom, yuntil - yfrom)
                    for (mx <- xfrom until xuntil) {
                        for (my <- yfrom until yuntil) {
                            matrix.base(mx - xfrom, my - yfrom) = base(mx, my)
                        }
                    }
                    result(y * nx + x) = matrix
                }
            }
            result
        }

        def +(b : Matrix) = {
            assert (base.size == b.base.size)
            val clone = base.clone()
            for (y <- 0 until base.height) {
                for (x <- 0 until base.width) {
                    clone(x, y) = clone(x, y) + b.base(x, y)
                }
            }
            new Matrix(clone)
        }

        def *(b : Matrix) : Matrix = {
            assert (base.size == b.base.size)
            if (base.size == (1, 1)) {
                return this * b.base(0, 0)
            }

            val a_blks = chop(2, 2)
            val b_blks = b.chop(2, 2)

            val adds = new Array[Matrix](2 * first_nadds)
            for (i <- 0 until first_nadds) {
                adds(i) = ??(a_blks) + ??(a_blks) * ??(-1, 0, 1)
            }
            for (i <- 0 until first_nadds) {
                adds(i) = ??(b_blks) + ??(b_blks) * ??(-1, 0, 1)
            }

            val multiplies = (for (i <- 0 until nmultiplies)
                yield (??(adds)) * (??(adds))).toArray

            val final_blks = new Array[Matrix](4)

            for (nadds <- 0 until final_nadds) {
                val idx = ??(final_blks.length)
                final_blks(idx) = final_blks(idx) +
                    // GOAL -- len(multiplies) must be optimized to 7
                    multiplies(??(multiplies.length)) * ??(-1, 0, 1)
            }

            new Matrix((final_blks(0).base hcat final_blks(1).base) vcat
                (final_blks(2).base hcat final_blks(3).base))
        }

        def *(b : Int) = new Matrix(base map ((x, y, v) => v * b))

        def this(width : Int, height : Int) =
            this(new Array2D[Int](width, height))
    }

    // FIXME FIXME FIXME FIXME FIXME FIXME
    // Matrix will have $outer$ bound, which means that the ??()'s
    // in there will call the wrong function.
    val tests = Array(
        // [1 2 ; 2 3] * [2 0; 0 2] = [2 4 ; 4 6]
            ( new Matrix(new Array2D(Array(Array(1, 2), Array(2, 3)))),
                new Matrix(new Array2D(Array(Array(2, 0), Array(0, 2)))),
                new Matrix(new Array2D(Array(Array(2, 4), Array(4, 6)))) )
        )

    def main(a : Matrix, b : Matrix, correct : Matrix) {
        assert (a * b == correct)
    }
}

object StrassenMultiplyTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => new StrassenMultiplySketch(6, 7, 12))
    }
}
