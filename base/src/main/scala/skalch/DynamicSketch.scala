package skalch

import sketch.dyn.main.old.{ScOldDynamicSketch, ScSynthesisMain, ScTestGenerator}
import sketch.dyn.constructs.ScConstructInfo
import sketch.util.DebugOut

/**
 * Dynamic sketching library
 * @author gatoatigrado (Nicholas Tung) [ntung at ntung]
 * @author Casey Rodarmor [casey at rodarmor]
 *
 * See examples in the "test" directory for usage.
 * All holes, oracles inputs, and input generators are created as the class
 * is instantiated. This introduces a bit of structure without excessive
 * overhead.
 */
abstract class DynamicSketch extends ScOldDynamicSketch {
    private class FreezableVector[T <: ScConstructInfo] extends java.util.Vector[T] {
        var frozen = false
        def +=(value : T) {
            assert(!frozen)
            super.add(value)
        }
        def length = super.size()
        def get_and_freeze() = {
            frozen = true
            // not sure why not Array[ScConstructInfo]()...
            super.toArray(new Array[ScConstructInfo](0))
        }
    }
    private[this] val __input_list = new FreezableVector[InputGenerator]()
    // more fields in ScDynamicSketch.java

    /** generate test data */
    abstract class TestGenerator extends ScTestGenerator {
        def put_default_input(v : Int) = put_input(default_input_gen, v)
    }
    object NullTestGenerator extends TestGenerator {
        def set() { }
        def tests() { test_case() }
    }

    // am I abusing the type inference system? If anything is ambiguous, please email...
    // NOTE - the "[[string]]" part of the annotations is what is currently recognized
    // from the compiler. This should be an associative recognition in the future.

    /** NOTE - description annotations are necessary to know how to complete the hole. */
    @DescriptionAnnotation("[[integer untilv hole]] basic hole")
    def ??(uid: Int, untilv: Int): Int =
        DynamicSketch.this.ctrl_conf.getDynamicValue(uid, untilv)

    @DescriptionAnnotation("[[object apply hole]] list select hole")
    def ??[T](uid : Int, list: List[T]) : T =
        list(DynamicSketch.this.ctrl_conf.getDynamicValue(uid, list.length))

    @DescriptionAnnotation("[[object apply hole]] array select hole")
    def ??[T](uid : Int, arr: Array[T]) : T =
        arr(DynamicSketch.this.ctrl_conf.getDynamicValue(uid, arr.length))




    @DescriptionAnnotation("[[boolean oracle]] boolean oracle")
    def !!(uid : Int) : Boolean =
        DynamicSketch.this.oracle_conf.dynamicNextValue(uid, 2) == 1

    @DescriptionAnnotation("[[integer untilv oracle]] basic oracle")
    def !!(uid: Int, untilv: Int): Int =
        DynamicSketch.this.oracle_conf.dynamicNextValue(uid, untilv)

    @DescriptionAnnotation("[[object apply oracle]] array select oracle")
    def !![T](uid : Int, arr: Array[T]) : T =
        arr(DynamicSketch.this.oracle_conf.dynamicNextValue(uid, arr.length))

    @DescriptionAnnotation("[[object apply oracle]] list select oracle")
    def !![T](uid : Int, list: Seq[T]) : T =
        list(DynamicSketch.this.oracle_conf.dynamicNextValue(uid, list.length))

    @DescriptionAnnotation("[[object apply oracle]] 2 value select oracle")
    def !![T](uid : Int, v1 : T, v2 : T) : T =
        DynamicSketch.this.oracle_conf.dynamicNextValue(uid, 2) match {
            case 0 => v1
            case 1 => v2
        }

    @DescriptionAnnotation("[[object apply oracle]] 3 value select oracle")
    def !![T](uid : Int, v1 : T, v2 : T, v3 : T) : T =
        DynamicSketch.this.oracle_conf.dynamicNextValue(uid, 3) match {
            case 0 => v1
            case 1 => v2
            case 2 => v3
        }



    @DescriptionAnnotation("[[integer untilv oracle]] basic oracle with debugging")
    def `!!d`(uid: Int, untilv: Int): Int = {
        import java.lang.Integer
        assert(untilv > 0, "sketch provided bad untilv, not greater than zero. untilv="
            + untilv)
        val rv = DynamicSketch.this.oracle_conf.dynamicNextValue(uid, untilv)
        skCompilerAssert(rv >= 0 && rv < untilv, "compiler returned bad result",
            "result", rv : Integer, "untilv", untilv : Integer)
        rv
    }

    @DescriptionAnnotation("[[object apply oracle]] array select oracle with debugging")
    def `!!d`[T](uid : Int, arr: Array[T]) : T = {
        import java.lang.Integer
        val untilv = arr.length
        assert(untilv > 0, "sketch provided bad untilv, not greater than zero. untilv="
            + untilv)
        val rv = DynamicSketch.this.oracle_conf.dynamicNextValue(uid, untilv)
        skCompilerAssert(rv >= 0 && rv < untilv, "compiler returned bad result",
            "result", rv : Integer, "untilv", untilv : Integer)
        arr(rv)
    }

    // === Inputs ===
    class InputGenerator(val untilv : Int) extends ScConstructInfo {
        val uid : Int = __input_list.length
        __input_list += this
        def apply() = DynamicSketch.this.input_conf.nextValue(uid)
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
    class NullTestGeneratorCls extends TestGenerator {
        def set() { put_default_input(0) }
        def tests() { test_case() }
    }
    def NullTestGenerator() = new NullTestGeneratorCls()



    // === Convenience shortcuts for generators ===
    // note that these don't allow any structure within the generator to be exploited.
    // atm. this isn't much, but when it makes sketching faster, warnings should
    // be issued.
    val default_input_gen = new InputGenerator(1 << 30)
    def next_int_input() : Int = default_input_gen()



    // === Utility functions ===
    // Most of these should be in ScDynamicSketch if possible, but some
    // constructs like "def" params are only available for Scala.
    def skdprint(x : => String) {
        if (debug_print_enable) {
            skdprint_backend(x)
        }
    }

    def skdprint_loc(x : => String) {
        if (debug_print_enable) {
            skdprint_location_backend(x)
        }
    }



    def get_input_info() = __input_list.get_and_freeze()
}

object synthesize {
    def apply(f : (() => DynamicSketch)) : AnyRef = {
        val synth = new ScSynthesisMain(f)
        synth.synthesize()
    }
}
