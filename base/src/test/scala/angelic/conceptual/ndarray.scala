/*
package angelic.conceptual

object ArrayUtil {
    def clone[T](x : Array[T]) = {
        val rv = new Array[T](x.length)
        Array.copy(x, 0, rv, 0, x.length)
        rv
    }
}

/** values stored as an array of rows */
class Array2D[T](val width : Int, val height : Int, var values : Array[Array[T]]) {
    assert(values.length == height, "wrong height of initial values")
    for (v <- values) { assert(v.length == width, "wrong width of initial values") }

    def size = (width, height)
    def apply(x : Int, y : Int) = (values(y))(x)
    def update(x : Int, y : Int, value : T) { (values(y))(x) = value; }

    def vcat(other : Array2D[T]) : Array2D[T] = {
        assert(width == other.width)
        val next_values = new Array[Array[T]](height + other.height)
        for (i <- 0 until height) { next_values(i) = ArrayUtil.clone(values(i)) }
        for (i <- 0 until other.height) { next_values(height + i) = ArrayUtil.clone(other.values(i)) }
        return new Array2D[T](width, height + other.height, next_values)
    }

    def hcat(other : Array2D[T]) : Array2D[T] = {
        assert(height == other.height)
        val next_values = new Array[Array[T]](height)
        val next_width = width + other.width
        for (i <- 0 until height) {
            next_values(i) = new Array[T](next_width)
            for (a <- 0 until width) { (next_values(i))(a) = values(i)(a) }
            for (a <- 0 until other.width) {
                (next_values(i))(a + width) = other.values(i)(a)
            }
        }
        new Array2D[T](width + other.width, height, next_values)
    }

    override def toString() = {
        "Array2D <<< " + ("" /: values)( (v1, v2) =>
            v1 + "\n" + ("" /: v2)(_ + ", " + _) ) + " >>>"
    }

    override def clone() : Array2D[T] = {
        val next_values = new Array[Array[T]](height)
        for (y <- 0 until height) {
            next_values(y) = ArrayUtil.clone(values(y))
        }
        new Array2D(width, height, next_values)
    }

    def map(f : ((Int, Int, T) => T)) = {
        val rv = clone()
        for (y <- 0 until height) {
            for (x <- 0 until width) {
                rv(x, y) = f(x, y, rv(x, y))
            }
        }
        rv
    }

    def this(width : Int, height : Int) = {
        this(width, height, {
            val values = new Array[Array[T]](height)
            for (i <- 0 until height) {
                values(i) = new Array[T](width)
            }
            values })
    }

    def this(width : Int, height : Int, initial : T) = {
        this(width, height, {
            val values = new Array[Array[T]](height)
            for (i <- 0 until height) {
                values(i) = new Array[T](width)
                for (c <- 0 until width) { (values(i))(c) = initial }
            }
            values })
    }

    def this(values : Array[Array[T]]) = this(values(0).length, values.length, values)
}
*/
