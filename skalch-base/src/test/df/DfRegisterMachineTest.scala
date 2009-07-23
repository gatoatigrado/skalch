package test.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

class DfRegisterMachineSketch() extends AbstractDfSketch() {
    class Register(var value : Int)
    val registers = (for (i <- 0 until 10) yield new Register(-1)).toArray
    var num_registers = -1

    override def df_init() {
        num_registers = 6
        skAddCost(num_registers)
    }

    def reset_registers() {
        var i = 0
        while (i < num_registers) {
            registers(i).value = !!(0, num_buckets - 1, num_buckets)
            i += 1
        }
    }

    def df_main() {
        skdprint_loc("df_main()")
        val num_steps = !!(num_buckets + 1)
        for (i <- (0 until num_buckets)) {
            buckets(i) match {
                case Red() =>
                    swap(i, registers(??(num_registers)).value)
                    registers(??(num_registers)).value += ??(2)
                    registers(??(num_registers)).value += ??(2)
                    registers(??(num_registers)).value += ??(2)
                case White() =>
                    registers(??(num_registers)).value += ??(2)
                    registers(??(num_registers)).value += ??(2)
                    registers(??(num_registers)).value += ??(2)
                case Blue() =>
                    registers(??(num_registers)).value += ??(2)
                    registers(??(num_registers)).value += ??(2)
                    registers(??(num_registers)).value += ??(2)
            }
            swap(i, !!(num_buckets))
            swap(!!(num_buckets), !!(num_buckets))
            skdprint(abbrev_str())
        }
    }
}

object DfRegisterMachineTest {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfRegisterMachineSketch())
    }
}
