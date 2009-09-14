
class NdArrayLibraryClass[T](val sz : Int) {
    type mytype = NdArrayLibraryClass[T]
    var dimensions : Array[Int] = null
    var size = 0
    var values = new Array[T](sz)
    var mult = 1
    val multipliers : Array[Int] = null

    def init_recursive() : List[Int] = {
        var dimensions = sz :: (values(0) match {
            case subarray : NdArrayLibraryClass[_] => subarray.init_recursive()
            case _ => Nil
        })
        multipliers = dimensions map { v => { val prev = mult; mult *= v; prev } }
        return dimensions
    }

    def apply(idx : Int*) : T = {
        var idxi = (0 /: (for ((m, v) <- (multipliers zip idx)) yield m * v))(_ + _)
        values(idxi)
    }

//     def cat(dim : Int, other : mytype) : mytype = {
//         for (idx in 0 until dimensions.length) {
//             assert ( (idx == dim) || (dimensions(idx) == other.dimensions(idx)) );
//         }
//         val next_dimensions = dimensions.toList.toArray
//         next_dimensions(dim) += other.dimensions(dim)
//     }

//     def this(dimensions : Int*) = this(dimensions.toArray)

//     def update(idxvalue : Int*) : T = {
//         val idxarr = idxvalue.subArray(0, len(idxvalue) - 1)
//         var idx = (0 /: (for ((m, v) <- (multipliers zip idxarr)) yield m * v))(_ + _)
//         values(idx) = idxvalue(len(idxvalue) - 1)
//     }
}

def twodim_array[T](width : Int, height : Int) = {
    val result = new NdArrayLibraryClass[NdArrayLibraryClass[T]](height)
    for (x <- 0 until height) {
        result.values(a) = new NdArrayLibraryClass[T](width)
    }
    result
}

println("hello_world")
val arr1 = twodim_array[Int](1, 2)
val arr2 = twodim_array[Int](2, 2)
