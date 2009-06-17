package test

import scala.collection.mutable.ListBuffer
import RomanNumerals._
import skalch.DynamicSketch
import sketch.dyn.BackendOptions
import sketch.util.DebugOut
import sketch.util._

object RomanNumerals1 extends DynamicSketch {
    val loopbnd = new BooleanOracle()
    val numeral_select = new ValueSelectOracle[RomanNumeral](num=7)

    def numeralOfNumber(n : Int) : List[RomanNumeral] = {
        val buf = new ListBuffer[RomanNumeral]
        var i = 0
        while (i < n && !loopbnd()) {
            buf += numeral_select(I(), V(), X(), L(), C(), D(), M())
            i += 1
        }
        DebugOut.print_mt("list", buf)
        return buf.toList
    }

    def dysketch_main() {
        val n = next_int_input()
        assert(numberOfNumeral(numeralOfNumber(n)) == n)
    }

    object ExhaustiveTestGenerator extends TestGenerator {
        // this is supposed to be expressive only, recover it with Java reflection if necessary
        def set(x : Int) {
            set_default_input(Array(x))
        }
        var decimal_number = 10
        def tests() {
            test_case(decimal_number : java.lang.Integer)
        }
    }

    object TestOptions extends CliOptGroup {
        add("--decimal_number", 1 : java.lang.Integer)
    }

    def main(args : Array[String])  = {
        val cmdopts = new sketch.util.CliParser(args)
        val be_opts = BackendOptions.create_and_parse(cmdopts)
        ExhaustiveTestGenerator.decimal_number = TestOptions.parse(cmdopts).int_("decimal_number")
        RomanNumerals1.synthesize_from_test(ExhaustiveTestGenerator, be_opts)
    }
}

/*
package choice.studies

import Choice._
import RomanNumerals._

 * Version 1
 * ---------
 * Correct, but just puts n Is together.
 * Sample trace (8): !!: Solution found after 1 trials in 00:00.17: true, I(), true, I(), true, I(), true, I(), true, I(), true, I(), true, I(), true, I()
object RomanNumerals1 {

  import scala.collection.mutable.ListBuffer

  def numeralOfNumber(n: Int): List[RomanNumeral] = {
    val buf = new ListBuffer[RomanNumeral]
    boundedWhile (n) (choiceBoolean()) {
      buf += !![RomanNumeral](I(), V(), X(), L(), C(), D(), M())
    }
    buf.toList
  }

  def check(n: Int): Unit = {
    val numeral = numeralOfNumber(n)
    val number = numberOfNumeral(numeral)
    assert(number == n)
  }

  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Please give me at least one argument, a number.")
      System.exit(2)
    }
    //args foreach { arg => Console.println(numberOfNumeral(numeralOfString(arg))) }
    args foreach { arg => run(check(Integer parseInt arg)) }
  }

}
*/
