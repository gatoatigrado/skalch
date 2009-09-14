
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
}

println("hello_world")
val arr1 = new Array2D[Int](1, 2, 1)
val arr2 = new Array2D[Int](2, 2, 0)
val arr3 = new Array2D[Int](2, 1, 3)
val arr4 = new Array2D[Int](2, 2, -1)
arr2(0, 0) = 1
arr2(1, 1) = 1
arr3(1, 0) = 4
arr4(1, 0) = 3
arr4(0, 1) = 13
arr4(1, 1) = 6
println("array 1", arr1)
println("array 2", arr2)
println("array 3", arr3)
println("array 4", arr4)
println("[array1, array2]", arr1 hcat arr2)
println("[array3; array4]", arr3 vcat arr4)
