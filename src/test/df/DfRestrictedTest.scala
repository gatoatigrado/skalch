package test.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfRestrictedSketch() extends AbstractDfSketch() {
    class Register(var value : Int)
    val registers = (for (i <- 0 until 10) yield new Register(-1)).toArray
    var num_registers = -1

    override def df_init() {
        num_registers = !!(6) + 1
        skAddCost(num_registers)
    }

    def reset_registers() {
        var i = 0
        while (i < num_registers) {
            registers(i).value = 0
            i += 1
        }
    }

    def read_reg(idx : Int) : Int = {
        val rv = registers(idx).value
        assert(idx < num_registers)
        assert(rv >= 0)
        rv
    }

    def df_main() {
        skdprint_loc("df_main()")
        reset_registers()
        for (i <- (0 until num_buckets)) {
            buckets(i) match {
                case Blue() =>
                    swap(read_reg(!!(num_registers)), i)
                case Red() =>
                    swap(read_reg(!!(num_registers)), i)
                case White() =>
                    swap(read_reg(!!(num_registers)), i)
            }
            skdprint(abbrev_str())
        }
    }
}

object DfRestrictedTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfRestrictedSketch())
    }
}
