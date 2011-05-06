package angelic.old.cossimple
import skalch.AngelicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

/** rewrite for Skalch + SKETCH integration */
class GaFindMeSketch() extends AngelicSketch {
    val tests = Array( 400000, 300000, 100000 )
    def main(value_to_match : Int) {
        val guess = !!(1000000)
        skdprint("input: " + value_to_match + ", value: " + guess)
        val difference = Math.abs(guess - value_to_match)
        addCost(difference)
        synthAssert(difference == 0)
        addCost(-1000000) // reward it for getting farther.
    }
}

object GaFindMeTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.addOpts(cmdopts)
        skalch.AngelicSketchSynthesize(() => new GaFindMeSketch())
    }
}
