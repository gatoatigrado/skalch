package skalch_old

import scala.collection.mutable.ListBuffer
import RomanNumerals._
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util._

class RomanNumerals1(val decimal_number : Int) extends DynamicSketch {
    def numeralOfNumber(n : Int) : List[RomanNumeral] = {
        val buf = new ListBuffer[RomanNumeral]
        var i = 0
        while (i < n && !(!!())) {
            buf += !!(List(I(), V(), X(), L(), C(), D(), M()))
            i += 1
        }
//         print("list", buf.toString)
        return buf.toList
    }

    def dysketch_main() = {
        val n = next_int_input()
        val encoded = numeralOfNumber(n)
        val decoded = numberOfNumeral(encoded)
//         print("n", n.toString, "encoded", encoded.toString, "decoded", decoded.toString)
        decoded == n
    }

    val test_generator = new TestGenerator {
        def set(x : Int) {
            put_default_input(x)
        }
        def tests() {
            test_case(decimal_number : java.lang.Integer)
        }
    }
}

object RomanNumeralsTest {
    object TestOptions extends cli.CliOptionGroup {
        addOption("decimal_number", 1 : java.lang.Integer, "which roman numeral to generate")
    }

    def main(args : Array[String])  = {
        DebugOut.todo("The RomanNumeralsTest currently doesn't decode roman",
            "numerals correctly; for example, it reports that \"iiv\" is 5.",
            "Please contact Nicholas Tung or Joel Galenson if you need it fixed.")
        val cmdopts = new cli.CliParser(args)
        BackendOptions.add_opts(cmdopts)
        val decimal_number = TestOptions.parse(cmdopts).long_("decimal_number").intValue
        skalch.synthesize(() => new RomanNumerals1(decimal_number))
    }
}
