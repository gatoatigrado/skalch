package angelic.simple

import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class NdArrayLibraryClass[T](dimensions : Int*) {
    var size = (1 /: dimensions)(_ * _)
    var values = new Array[T](1)
    var mult = 1
    val multipliers = dimensions map { v => { val prev = mult; mult *= v; prev } }

    def apply(idx : Int*) : T = {
        var idx = (0 /: (for ((m, v) <- (multipliers zip idx)) yield m * v))(_ + _)
        values(idx)
    }

    def cat(dimension : Int, other : NdArrayLibraryClass[T]) : NdArrayLibraryClass[T] = {
    }

//     def update(idxvalue : Int*) : T = {
//         val idxarr = idxvalue.subArray(0, len(idxvalue) - 1)
//         var idx = (0 /: (for ((m, v) <- (multipliers zip idxarr)) yield m * v))(_ + _)
//         values(idx) = idxvalue(len(idxvalue) - 1)
//     }
}

class StrassenMultiplySketch() extends AngelicSketch {
    class Matrix (width : Int, height : Int) extends NdArrayLibraryClass[Int](width, height) {
        def chop(nx : Int, ny : Int)  : Array[Matrix] = {
            result := Array[Matrix](nx * ny)
            for (x <- 0 until nx) {
                for (y <- 0 until ny) {
                    val xfrom = (x * width) / nx
                    val xuntil = ((x + 1) * width) / nx
                    val yfrom = (y * width) / ny
                    val yuntil = ((y + 1) * width) / ny
                    val matrix = Matrix(xuntil - xfrom, yuntil - yfrom)
                    for (mx <- xfrom until xuntil) {
                        for (my <- yfrom until yuntil) {
                            matrix(mx - xfrom, my - yfrom) = apply(mx, my)
                        }
                    }
                    result(y * nx + x) = matrix
                }
            }
            result
        }

        def +(b : Matrix) = {
            assert (size == b.size)
            Matrix(width, height, (values zip b.values) map (_ + _))
        }

        def *(b : Matrix) {
            assert (size == b.size)
            if (size == (1, 1)) {
                return values[0] * b.values[0]
            }

            a_blks = chop(2, 2)
            b_blks = b.chop(2, 2)

            adds := { (a_blks) + (a_blks) * ??(-1, 1) }
                and { (b_blks) + (b_blks) * ??(-1, 1) }
            multiplies := { ??(adds) * ??(adds) }
            final_blks = (for (i <- 0 until 4)) { yield
                Matrix(width/2, height/2) }).toArray
            for (nadds <- 0 until 11) {
                idx = ??(len(second_adds))
                final_blks(idx) = second_adds(idx) +
                    ??(-1, 1) * mutliplies(??(len(multiplies)))
            }
            join(??, final_blks)
        }

        def *(b : Int) =
            Matrix(width, height, values map (b * _))
        }

        def this(width : Int, height : Int) =
            this(width, height, new Array[Int](width * height))
    }

    def hjoin(a : Matrix*) = {
        result = Matrix(a(0).width * a.length, a(0).height)
        for (i <- 0 until a.length) {
            m = a(i)
            for (v <- m.values) {
                result.values(idx) = v
                idx += 1
            }
        }
        result
    }

    def join(nx : Int, a : Matrix*) {
        ny = a.length / nx
        assert a.length % nx == 0
        sz = a(0).size
        width, height = nx * sz(0), ny * sz(1)
        result = Matrix(width, height)
        idx = 0
        for (y <- 0 until ny) {
            for (my <- 0 until sz(1)) {
                for (x <- 0 until nx) {
                    m = a(y * nx + x)
                    for (mx <- 0 until mx.width) {
                        result(idx) = m(x, y)
                        idx += 1
                    }
                }
            }
        }
        result
    }

    val tests = Array(
        ( Matrix(2, 2, Array(1, 0, 0, 1),
            Matrix(2, 2, Array(1, 0, 0, 0),
            Matrix(2, 2, Array(1, 0, 0, 0) )
        )

    def main(a : Matrix, b : Matrix, correct : Matrix) {
        assert (a * b == correct)
    }
}

object StrassenMultiplyTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
//         skalch.AngelicSketchSynthesize(() => new StrassenMultiplySketch())
    }
}
