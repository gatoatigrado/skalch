package test.skalch_old.df

import sketch.dyn.BackendOptions
import sketch.util._
import DebugOut.print

/**
 * Difficult test for developing the GA.
 * Currently takes backtracking search 9 659 860 iterations
 * in 37 seconds to find one solution
 * b fsc '/(Dynamic|DfRegister)' run_app=test.df.DfRegisterTestDifficult run_opt_list --ui_no_gui --sy_num_solutions 1
 * runs / sec: 261 225.56
 */
class DfRegisterSketch() extends AbstractDfSketch() {
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
            registers(i).value = 0
            i += 1
        }
    }

    def df_main() {
        skdprint_loc("df_main()")
        reset_registers()
        var i = 0
        while (i < num_buckets) {
            var color = buckets(i).order
            buckets(i) match {
                case Red() =>
                    swap(registers(??(num_registers)).value, i)
                case White() =>
                    swap(registers(??(num_registers)).value, i)
                case Blue() =>
                    swap(registers(??(num_registers)).value, i)
            }

            if (!!()) {
                val regidx0 = !!(num_registers)
                val regidx1 = !!(num_registers)
                skdprint("additional swap: " + regidx0 + ", " + regidx1)
                swap(registers(regidx0).value, registers(regidx1).value)
                skAddCost(1)
            }
            //synthAssertTerminal(isCorrect(i + 1))

            while (color < 3) {
                registers(color).value += 1
                color += 1
            }

            skdprint(abbrev_str())
            i += 1
        }
    }
}

object DfRegisterTestDifficult {
    def main(args : Array[String]) = {
        val cmdopts = new cli.CliParser(args)
        AbstractDfOptions.result = AbstractDfOptions.parse(cmdopts)
        BackendOptions.add_opts(cmdopts)
        skalch.synthesize(() => new DfRegisterSketch())
    }
}
