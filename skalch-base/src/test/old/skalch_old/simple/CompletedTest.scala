package skalch_old.simple
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

class CompletedSketch() extends DynamicSketch {

    def dysketch_main() = {
        true
    }

    val test_generator = new TestGenerator {
        def set(x : Int, v : Boolean) {
            put_default_input(x)
        }

        def tests() {
            for (ctr <- 0 until 4)
                test_case(new java.lang.Integer(ctr), new java.lang.Boolean(false))
        }
    }
}

object CompletedTest {
    def main(args: Array[String])  = {
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new CompletedSketch())
    }
}
