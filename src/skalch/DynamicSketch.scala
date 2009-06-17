package skalch

import sketch.dyn._
import sketch.util.DebugOut
import sketch.util.OptionResult

/**
 * Dynamic sketching library
 * @author gatoatigrado (Nicholas Tung) [ntung at ntung]
 *
 * See examples in the "test" directory for usage.
 * All holes, oracles inputs, and input generators are created as the class
 * is instantiated. This introduces a bit of structure without excessive
 * overhead.
 */
abstract class DynamicSketch extends ScDynamicSketch {
    private[this] var hole_list : List[Hole] = List()
    private[this] var input_gen_list : List[InputGenerator] = List()
    private[this] var oracle_input_list : List[OracleInput] = List()
    // more fields in ScDynamicSketch.java

    /** generate test data */
    abstract class TestGenerator extends ScTestGenerator {
        def set_input(input : InputGenerator, v : Array[Int]) = set_uid(input.uid, v)
        def set_default_input(v : Array[Int]) = set_input(default_input_gen, v)
    }
    object NullTestGenerator extends TestGenerator {
        def set() { }
        def tests() { test_case() }
    }



    // === Holes ===
    class Hole(val untilv : Int) extends ScConstructInfo {
       var uid : Int = hole_list.length
       hole_list ::= this
       def apply() = DynamicSketch.this.ctrl_values(uid).get_value() // hopefully wickedly fast
       override def toString() = "Hole[uid = " + uid + ", untilv = " + untilv + ", cv = value]"
    }
    class NegHole(val mag_untilv : Int) extends Hole(2 * mag_untilv - 1) {
        override def apply() = {
            val v = super.apply()
            val posvalue = (v >> 1) - (v & 1)
            if ((v & 1) == 1) (-posvalue) else posvalue
        }
    }
    class MinMaxHole(val min_incl : Int, val max_incl : Int
        ) extends Hole(max_incl - min_incl + 1)
    {
        override def apply() = super.apply() + min_incl
    }
    class ValueSelectHole[T](num : Int) extends Hole(num) {
        def apply(arr : T*) : T = { assert(arr.length == num); arr(super.apply()) }
    }
    def hole_array(num: Int, untilv: Int) =
        (for (i <- 0 until num) yield new Hole(untilv)).toArray



    // === Inputs ===
    class InputGenerator(val untilv : Int) extends ScConstructInfo {
        val uid : Int = input_gen_list.length
        input_gen_list ::= this
        def apply() = DynamicSketch.this.input_backend(uid).next_value()
    }
    class ArrayInput(val len_untilv : Int, val value_untilv : Int) {
        val len_generator = new InputGenerator(len_untilv)
        val value_generator = new InputGenerator(value_untilv)
        def next_array() : Array[Int] = {
            val n = len_generator()
            val result = new Array[Int](n)
            for (i <- 0 until n) result(i) = value_generator()
            return result
        }
    }



    // === Oracles inputs ===
    class OracleInput(val untilv : Int) extends ScConstructInfo {
        val uid : Int = oracle_input_list.length
        oracle_input_list ::= this
        def apply() : Int = DynamicSketch.this.oracle_input_backend(uid).next_value()
    }
    class BooleanOracle {
        val oracle = new OracleInput(2)
        def apply() : Boolean = (oracle.apply() == 1)
    }
    class ValueSelectOracle[T](num : Int) extends OracleInput(num) {
        def apply(arr : T*) : T = { assert(arr.length == num); arr(super.apply()) }
    }



    // === Convenience shortcuts for generators ===
    // note that these don't allow any structure within the generator to be exploited.
    // atm. this isn't much, but when it makes sketching faster, warnings shoudl
    // be issued.
    val default_input_gen = new InputGenerator(1 << 30)
    def next_int_input() : Int = default_input_gen()



    def get_hole_info() = hole_list.toArray
    def get_input_info() = input_gen_list.toArray
    def get_oracle_input_list() = oracle_input_list.toArray

    // === Main solver code ===
    def synthesize_from_test(tg : TestGenerator, be_opts : OptionResult) {
        DebugOut.todo("query user for test cases. running all for now.")
        val synth = new ScSynthesis(this.getClass, be_opts)
        synth.synthesize(tg)
    }

    /** null generator when test inputs are not explicit */
    def synthesize() {
        println("please fix synthesize() or use synthesize_from_test, sorry!")
        //synthesize_from_test(NullTestGenerato, null)
    }
}
