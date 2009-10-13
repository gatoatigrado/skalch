package angelic.simple
import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

/** rewrite for Skalch + SKETCH integration */
class SugaredSketch() extends AngelicSketch {
    val tests = Array( () )
    def main() {
        synthAssertTerminal(??(List("a", "b", "c")) == "c")
        synthAssertTerminal(??(100) == 63)
        for (i <- 0 until 10) {
            synthAssertTerminal(!!(10) == i);
        }
    }
}

object SugaredTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.AngelicSketchSynthesize(() => new SugaredSketch())
    }
}
