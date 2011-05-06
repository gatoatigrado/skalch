package angelic.old.simple
import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

/** rewrite for Skalch + SKETCH integration */
class SugaredSketch() extends AngelicSketch {
    val tests = Array( () )
    def main() {
        synthAssert(??(List("a", "b", "c")) == "c")
        synthAssert(??(100) == 63)
        for (i <- 0 until 10) {
            synthAssert(!!(10) == i);
        }
    }
}

object SugaredTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => new SugaredSketch())
    }
}
