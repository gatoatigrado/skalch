package angelic.simple
import skalch.{AngelicSketch, SkalchIgnoreClass}
import sketch.util._

class MySketch() extends AngelicSketch {
    def main() {
        var x = 0
        if (x > 3)
            x = 4
        assert (x == 0)
        x = 2
        assert(x + (?? : Int) == 4)
    }
}
