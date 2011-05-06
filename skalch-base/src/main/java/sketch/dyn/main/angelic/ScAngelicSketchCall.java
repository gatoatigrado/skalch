package sketch.dyn.main.angelic;

import static sketch.util.DebugOut.assertFalse;
import static sketch.util.DebugOut.assertSlow;
import static sketch.util.DebugOut.print_exception;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Vector;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.ui.queues.QueueIterator;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.util.DebugOut;

public class ScAngelicSketchCall implements ScDynamicSketchCall<ScAngelicSketchBase> {
    /** array of tuples */
    public final Object[][] testCases;
    public final ScAngelicSketchBase sketch;
    public final Method mainMethod;

    @SuppressWarnings("unchecked")
    public ScAngelicSketchCall(ScAngelicSketchBase sketch) {
        this.sketch = sketch;
        Method mainMethod = null;
        for (Method m : sketch.getClass().getMethods()) {
            if (m.getName().equals("main")) {
                mainMethod = m;
            }
        }
        this.mainMethod = mainMethod;
        if (mainMethod == null) {
            assertFalse("main method null.");
        }
        int nparams = mainMethod.getParameterTypes().length;
        Object[] tupleArr = null;
        try {
            Object testCasesObj = sketch.getClass().getMethod("tests").invoke(sketch);
            if (testCasesObj instanceof Object[]) {
                tupleArr = (Object[]) testCasesObj;
            } else if (testCasesObj instanceof int[]) {
                int[] intArr = (int[]) testCasesObj;
                tupleArr = new Object[intArr.length];
                for (int a = 0; a < intArr.length; a++) {
                    tupleArr[a] = new Integer(intArr[a]);
                }
            } else if (testCasesObj instanceof long[]) {
                long[] longArr = (long[]) testCasesObj;
                tupleArr = new Object[longArr.length];
                for (int a = 0; a < longArr.length; a++) {
                    tupleArr[a] = new Long(longArr[a]);
                }
            } else if (testCasesObj instanceof float[]) {
                float[] floatArr = (float[]) testCasesObj;
                tupleArr = new Object[floatArr.length];
                for (int a = 0; a < floatArr.length; a++) {
                    tupleArr[a] = new Float(floatArr[a]);
                }
            } else if (testCasesObj instanceof boolean[]) {
                boolean[] booleanArr = (boolean[]) testCasesObj;
                tupleArr = new Object[booleanArr.length];
                for (int a = 0; a < booleanArr.length; a++) {
                    tupleArr[a] = new Boolean(booleanArr[a]);
                }
            } else if (testCasesObj instanceof double[]) {
                double[] doubleArr = (double[]) testCasesObj;
                tupleArr = new Object[doubleArr.length];
                for (int a = 0; a < doubleArr.length; a++) {
                    tupleArr[a] = new Double(doubleArr[a]);
                }
            } else {
                assertFalse("ScAngelicSketchCall -- don't know what to do with "
                        + "test cases object", testCasesObj);
            }
        } catch (Exception e) {
            DebugOut.print_exception("requesting tests variable from sketch", e);
            assertFalse();
        }
        testCases = new Object[tupleArr.length][];
        // for i in $(seq 2 15); do echo
        // "} else if (elt instanceof scala.Tuple${i}) {test_cases[a] = arr(";
        // for c in $(seq 1 "$(($i - 1))"); do echo
        // "((scala.Tuple${i})elt)._${c}(),"; done; echo
        // "((scala.Tuple${i})elt)._${i}());"; echo
        // "assertSlow(nparams == ${i}, \"got ${i} parameters, expected\", nparams);";
        // done | xc
        for (int a = 0; a < tupleArr.length; a++) {
            Object elt = tupleArr[a];
            if (elt instanceof scala.runtime.BoxedUnit) {
                testCases[a] = arr();
                assertSlow(nparams == 0, "got 0 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple2) {
                testCases[a] = arr(((scala.Tuple2) elt)._1(), ((scala.Tuple2) elt)._2());
                assertSlow(nparams == 2, "got 2 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple3) {
                testCases[a] =
                        arr(((scala.Tuple3) elt)._1(), ((scala.Tuple3) elt)._2(),
                                ((scala.Tuple3) elt)._3());
                assertSlow(nparams == 3, "got 3 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple4) {
                testCases[a] =
                        arr(((scala.Tuple4) elt)._1(), ((scala.Tuple4) elt)._2(),
                                ((scala.Tuple4) elt)._3(), ((scala.Tuple4) elt)._4());
                assertSlow(nparams == 4, "got 4 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple5) {
                testCases[a] =
                        arr(((scala.Tuple5) elt)._1(), ((scala.Tuple5) elt)._2(),
                                ((scala.Tuple5) elt)._3(), ((scala.Tuple5) elt)._4(),
                                ((scala.Tuple5) elt)._5());
                assertSlow(nparams == 5, "got 5 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple6) {
                testCases[a] =
                        arr(((scala.Tuple6) elt)._1(), ((scala.Tuple6) elt)._2(),
                                ((scala.Tuple6) elt)._3(), ((scala.Tuple6) elt)._4(),
                                ((scala.Tuple6) elt)._5(), ((scala.Tuple6) elt)._6());
                assertSlow(nparams == 6, "got 6 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple7) {
                testCases[a] =
                        arr(((scala.Tuple7) elt)._1(), ((scala.Tuple7) elt)._2(),
                                ((scala.Tuple7) elt)._3(), ((scala.Tuple7) elt)._4(),
                                ((scala.Tuple7) elt)._5(), ((scala.Tuple7) elt)._6(),
                                ((scala.Tuple7) elt)._7());
                assertSlow(nparams == 7, "got 7 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple8) {
                testCases[a] =
                        arr(((scala.Tuple8) elt)._1(), ((scala.Tuple8) elt)._2(),
                                ((scala.Tuple8) elt)._3(), ((scala.Tuple8) elt)._4(),
                                ((scala.Tuple8) elt)._5(), ((scala.Tuple8) elt)._6(),
                                ((scala.Tuple8) elt)._7(), ((scala.Tuple8) elt)._8());
                assertSlow(nparams == 8, "got 8 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple9) {
                testCases[a] =
                        arr(((scala.Tuple9) elt)._1(), ((scala.Tuple9) elt)._2(),
                                ((scala.Tuple9) elt)._3(), ((scala.Tuple9) elt)._4(),
                                ((scala.Tuple9) elt)._5(), ((scala.Tuple9) elt)._6(),
                                ((scala.Tuple9) elt)._7(), ((scala.Tuple9) elt)._8(),
                                ((scala.Tuple9) elt)._9());
                assertSlow(nparams == 9, "got 9 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple10) {
                testCases[a] =
                        arr(((scala.Tuple10) elt)._1(), ((scala.Tuple10) elt)._2(),
                                ((scala.Tuple10) elt)._3(), ((scala.Tuple10) elt)._4(),
                                ((scala.Tuple10) elt)._5(), ((scala.Tuple10) elt)._6(),
                                ((scala.Tuple10) elt)._7(), ((scala.Tuple10) elt)._8(),
                                ((scala.Tuple10) elt)._9(), ((scala.Tuple10) elt)._10());
                assertSlow(nparams == 10, "got 10 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple11) {
                testCases[a] =
                        arr(((scala.Tuple11) elt)._1(), ((scala.Tuple11) elt)._2(),
                                ((scala.Tuple11) elt)._3(), ((scala.Tuple11) elt)._4(),
                                ((scala.Tuple11) elt)._5(), ((scala.Tuple11) elt)._6(),
                                ((scala.Tuple11) elt)._7(), ((scala.Tuple11) elt)._8(),
                                ((scala.Tuple11) elt)._9(), ((scala.Tuple11) elt)._10(),
                                ((scala.Tuple11) elt)._11());
                assertSlow(nparams == 11, "got 11 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple12) {
                testCases[a] =
                        arr(((scala.Tuple12) elt)._1(), ((scala.Tuple12) elt)._2(),
                                ((scala.Tuple12) elt)._3(), ((scala.Tuple12) elt)._4(),
                                ((scala.Tuple12) elt)._5(), ((scala.Tuple12) elt)._6(),
                                ((scala.Tuple12) elt)._7(), ((scala.Tuple12) elt)._8(),
                                ((scala.Tuple12) elt)._9(), ((scala.Tuple12) elt)._10(),
                                ((scala.Tuple12) elt)._11(), ((scala.Tuple12) elt)._12());
                assertSlow(nparams == 12, "got 12 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple13) {
                testCases[a] =
                        arr(((scala.Tuple13) elt)._1(), ((scala.Tuple13) elt)._2(),
                                ((scala.Tuple13) elt)._3(), ((scala.Tuple13) elt)._4(),
                                ((scala.Tuple13) elt)._5(), ((scala.Tuple13) elt)._6(),
                                ((scala.Tuple13) elt)._7(), ((scala.Tuple13) elt)._8(),
                                ((scala.Tuple13) elt)._9(), ((scala.Tuple13) elt)._10(),
                                ((scala.Tuple13) elt)._11(), ((scala.Tuple13) elt)._12(),
                                ((scala.Tuple13) elt)._13());
                assertSlow(nparams == 13, "got 13 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple14) {
                testCases[a] =
                        arr(((scala.Tuple14) elt)._1(), ((scala.Tuple14) elt)._2(),
                                ((scala.Tuple14) elt)._3(), ((scala.Tuple14) elt)._4(),
                                ((scala.Tuple14) elt)._5(), ((scala.Tuple14) elt)._6(),
                                ((scala.Tuple14) elt)._7(), ((scala.Tuple14) elt)._8(),
                                ((scala.Tuple14) elt)._9(), ((scala.Tuple14) elt)._10(),
                                ((scala.Tuple14) elt)._11(), ((scala.Tuple14) elt)._12(),
                                ((scala.Tuple14) elt)._13(), ((scala.Tuple14) elt)._14());
                assertSlow(nparams == 14, "got 14 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple15) {
                testCases[a] =
                        arr(((scala.Tuple15) elt)._1(), ((scala.Tuple15) elt)._2(),
                                ((scala.Tuple15) elt)._3(), ((scala.Tuple15) elt)._4(),
                                ((scala.Tuple15) elt)._5(), ((scala.Tuple15) elt)._6(),
                                ((scala.Tuple15) elt)._7(), ((scala.Tuple15) elt)._8(),
                                ((scala.Tuple15) elt)._9(), ((scala.Tuple15) elt)._10(),
                                ((scala.Tuple15) elt)._11(), ((scala.Tuple15) elt)._12(),
                                ((scala.Tuple15) elt)._13(), ((scala.Tuple15) elt)._14(),
                                ((scala.Tuple15) elt)._15());
                assertSlow(nparams == 15, "got 15 parameters, expected", nparams);
            } else {
                testCases[a] = arr(elt);
                assertSlow(nparams == 1, "got 1 parameters, expected", nparams);
            }
        }
    }

    public static Object[] arr(Object... elts) {
        return elts;
    }

    public int getNumCounterexamples() {
        return testCases.length;
    }

    public void initializeBeforeAllTests(ScCtrlConf ctrlConf, ScInputConf oracleConf,
            QueueIterator queueIterator)
    {
        sketch.solutionCost = 0;
        sketch.numAssertsPassed = 0;
        sketch.ctrlConf = ctrlConf;
        sketch.oracleConf = oracleConf;
        sketch.queueIterator = queueIterator;
    }

    public boolean runTest(int idx) {
        try {
            mainMethod.invoke(sketch, testCases[idx]);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            assertFalse("run test");
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            assertFalse("run test");
        } catch (InvocationTargetException e) {
            Throwable exception = e.getCause();
            if (exception instanceof RuntimeException) {
                if (sketch.debugPrintEnable) {
                    sketch.printBackend(exception.toString(), true, null);
                    StackTraceElement[] stackTrace = exception.getStackTrace();
                    for (StackTraceElement element : stackTrace) {
                        sketch.printBackend(element.toString(), true, null);
                    }
                    // sketch.skdprintBackend(exception.getStackTrace().toString());
                }
                sketch.synthAssert(false);
                // throw ((RuntimeException) exception);
            } else {
                print_exception("exception while executing sketch", e);
                assertFalse();
            }
        }
        return true;
    }

    public int getSolutionCost() {
        return sketch.solutionCost;
    }

    public ScAngelicSketchBase getSketch() {
        return sketch;
    }

    public ScConstructValueString getHoleValueString(int uid)
            throws ScNoValueStringException
    {
        return sketch.ctrlConf.getValueString(uid);
    }

    public Vector<ScConstructValueString> getOracleValueString(int uid)
            throws ScNoValueStringException
    {
        return sketch.oracleConf.getValueString(uid);
    }
}
