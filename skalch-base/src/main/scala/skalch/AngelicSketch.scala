package skalch

import sketch.util.DebugOut

/**
 * New dynamic sketching base class. This class supports the oracle,
 * and relies on a "tests" variable to be set.
 * See examples in the "test" directory for usage.
 * NOTE - more API functions in ScAngelicSketchBase
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
abstract class AngelicSketch extends sketch.dyn.main.angelic.ScAngelicSketchBase {
    // NOTE - the "[[string]]" part of the annotations is what is currently recognized
    // from the compiler. This should be an associative recognition in the future.

    /** NOTE - description annotations are necessary to know how to complete the hole. */
    @DescriptionAnnotation("[[integer untilv hole]] basic hole")
    def ??(uid: Int, untilv: Int): Int = ctrlConf.getDynamicValue(uid, untilv)

    @DescriptionAnnotation("[[object apply hole]] sequence select hole")
    def ??[T](uid: Int, list: Seq[T]): T =
        list(ctrlConf.getDynamicValue(uid, list.length))

    @DescriptionAnnotation("[[object apply hole]] array select hole")
    def ??[T](uid: Int, arr: Array[T]): T =
        arr(ctrlConf.getDynamicValue(uid, arr.length))

    @DescriptionAnnotation("[[list select hole]] varargs select hole")
    def ??[T](uid: Int, first: T, second: T, values: T*): T = {
        ctrlConf.getDynamicValue(uid, values.length + 2) match {
            case 0 => first
            case 1 => second
            case other => values(other - 2)
        }
    }

    @DescriptionAnnotation("[[boolean oracle]] boolean oracle")
    def !!(uid: Int): Boolean = oracleConf.dynamicNextValue(uid, 2) == 1

    @DescriptionAnnotation("[[integer untilv oracle]] basic oracle")
    def !!(uid: Int, untilv: Int): Int = oracleConf.dynamicNextValue(uid, untilv)

    @DescriptionAnnotation("[[object apply oracle]] array select oracle")
    def !![T](uid: Int, arr: Array[T]): T =
        arr(oracleConf.dynamicNextValue(uid, arr.length))

    @DescriptionAnnotation("[[object apply oracle]] sequence select oracle")
    def !![T](uid: Int, list: Seq[T]): T =
        list(oracleConf.dynamicNextValue(uid, list.length))

    @DescriptionAnnotation("[[object apply oracle]] 2 value select oracle")
    def !![T](uid: Int, v1: T, v2: T): T =
        oracleConf.dynamicNextValue(uid, 2) match {
            case 0 => v1
            case 1 => v2
        }

    @DescriptionAnnotation("[[object apply oracle]] 3 value select oracle")
    def !![T](uid: Int, v1: T, v2: T, v3: T): T =
        oracleConf.dynamicNextValue(uid, 3) match {
            case 0 => v1
            case 1 => v2
            case 2 => v3
        }

    @DescriptionAnnotation("[[integer untilv oracle]] basic oracle with debugging")
    def `!!d`(uid: Int, untilv: Int): Int = {
        import java.lang.Integer
        assert(untilv > 0, "sketch provided bad untilv, not greater than zero. untilv="
            + untilv)
        val rv = oracleConf.dynamicNextValue(uid, untilv)
        compilerAssert(rv >= 0 && rv < untilv, "compiler returned bad result",
            "result", rv: Integer, "untilv", untilv: Integer)
        rv
    }

    @DescriptionAnnotation("[[object apply oracle]] array select oracle with debugging")
    def `!!d`[T](uid: Int, arr: Array[T]): T = {
        import java.lang.Integer
        val untilv = arr.length
        assert(untilv > 0, "sketch provided bad untilv, not greater than zero. untilv="
            + untilv)
        val rv = oracleConf.dynamicNextValue(uid, untilv)
        compilerAssert(rv >= 0 && rv < untilv, "compiler returned bad result",
            "result", rv: Integer, "untilv", untilv: Integer)
        arr(rv)
    }

    // === Utility functions ===
    // Most of these should be in ScDynamicSketch if possible, but some
    // constructs like "def" params are only available for Scala.
    def print(x: => String) {
        if (debugPrintEnable) {
            printBackend(x, false, null)
        }
    }

    def print(x: => String, c: => java.awt.Color) {
        if (debugPrintEnable) {
            printBackend(x, false, c)
        }
    }

    def println(x: => String) {
        if (debugPrintEnable) {
            printBackend(x, true, null)
        }
    }

    def println(x: => String, c: => java.awt.Color) {
        if (debugPrintEnable) {
            printBackend(x, true, c)
        }
    }
    
    def print_loc(x: => String) {
        if (debugPrintEnable) {
            printLocationBackend(x)
        }
    }

    def last_angel_color(): java.awt.Color = {
        oracleConf.getLastAngelColor()
    }

    def put[T](x: T): T = {
        put(0, x)
    }

    def put[T](queueNum: Int, x: T): T = {
        if (debugPrintEnable) {
            queuePutBackend(queueNum, x)
        }
        x
    }

    def check[T](x: T): T = {
        check(0, x)
    }

    def check[T](queueNum: Int, x: T): T = {
        queueCheckBackend(queueNum, x, debugPrintEnable)
        x
    }

    def put_check[T](x: T): T = {
        put_check(0, x)
    }

    def put_check[T](queueNum: Int, x: T): T = {
        put(queueNum, x)
        check(queueNum, x)
        x
    }

    // === Utility functions ===
    // Most of these should be in ScDynamicSketch if possible, but some
    // constructs like "def" params are only available for Scala.

    @deprecated
    def skdprint(x: => String) {
        if (debugPrintEnable) {
            printBackend(x, true, null)
        }
    }

    @deprecated
    def skdprint(x: => String, c: => java.awt.Color) {
        if (debugPrintEnable) {
            printBackend(x, true, c)
        }
    }

    @deprecated
    def skdprint_loc(x: => String) {
        if (debugPrintEnable) {
            printLocationBackend(x)
        }
    }

    @deprecated
    def sklast_angel_color(): java.awt.Color = {
        oracleConf.getLastAngelColor()
    }
    
    @deprecated
    def skput[T](x: T): T = {
        put(0, x)
    }

    @deprecated
    def skput[T](queueNum: Int, x: T): T = {
        if (debugPrintEnable) {
            queuePutBackend(queueNum, x)
        }
        x
    }

    @deprecated
    def skcheck[T](x: T): T = {
        check(0, x)
    }

    @deprecated
    def skcheck[T](queueNum: Int, x: T): T = {
        queueCheckBackend(queueNum, x, debugPrintEnable)
        x
    }

    @deprecated
    def skput_check[T](x: T): T = {
        put_check(0, x)
    }

    @deprecated
    def skput_check[T](queueNum: Int, x: T): T = {
        put(queueNum, x)
        check(queueNum, x)
        x
    }
}

object AngelicSketchSynthesize {
    def apply(f: (() => AngelicSketch)): AnyRef = {
        val synth = new sketch.dyn.main.angelic.ScAngelicSynthesisMain(f)
        synth.synthesize()
    }
}
