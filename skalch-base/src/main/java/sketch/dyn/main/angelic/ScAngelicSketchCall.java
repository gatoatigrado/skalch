package sketch.dyn.main.angelic;

import static sketch.util.DebugOut.assertFalse;
import static sketch.util.DebugOut.assertSlow;
import static sketch.util.DebugOut.print_exception;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.util.DebugOut;

public class ScAngelicSketchCall implements
        ScDynamicSketchCall<ScAngelicSketchBase>
{
    /** array of tuples */
    public final Object[][] test_cases;
    public final ScAngelicSketchBase sketch;
    public final Method main_method;

    @SuppressWarnings("unchecked")
    public ScAngelicSketchCall(ScAngelicSketchBase sketch) {
        this.sketch = sketch;
        Method main_method = null;
        for (Method m : sketch.getClass().getMethods()) {
            if (m.getName().equals("main")) {
                main_method = m;
            }
        }
        this.main_method = main_method;
        if (main_method == null) {
            assertFalse("main method null.");
        }
        int nparams = main_method.getParameterTypes().length;
        Object[] tuple_arr = null;
        try {
            tuple_arr =
                    (Object[]) sketch.getClass().getMethod("tests").invoke(
                            sketch);
        } catch (Exception e) {
            DebugOut
                    .print_exception("requesting tests variable from sketch", e);
            assertFalse();
        }
        test_cases = new Object[tuple_arr.length][];
        // for i in $(seq 2 15); do echo
        // "} else if (elt instanceof scala.Tuple${i}) {test_cases[a] = arr(";
        // for c in $(seq 1 "$(($i - 1))"); do echo
        // "((scala.Tuple${i})elt)._${c}(),"; done; echo
        // "((scala.Tuple${i})elt)._${i}());"; echo
        // "assertSlow(nparams == ${i}, \"got ${i} parameters, expected\", nparams);";
        // done | xc
        for (int a = 0; a < tuple_arr.length; a++) {
            Object elt = tuple_arr[a];
            if (elt instanceof scala.runtime.BoxedUnit) {
                test_cases[a] = arr();
                assertSlow(nparams == 0, "got 0 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple2) {
                test_cases[a] =
                        arr(((scala.Tuple2) elt)._1(), ((scala.Tuple2) elt)
                                ._2());
                assertSlow(nparams == 2, "got 2 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple3) {
                test_cases[a] =
                        arr(((scala.Tuple3) elt)._1(), ((scala.Tuple3) elt)
                                ._2(), ((scala.Tuple3) elt)._3());
                assertSlow(nparams == 3, "got 3 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple4) {
                test_cases[a] =
                        arr(((scala.Tuple4) elt)._1(), ((scala.Tuple4) elt)
                                ._2(), ((scala.Tuple4) elt)._3(),
                                ((scala.Tuple4) elt)._4());
                assertSlow(nparams == 4, "got 4 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple5) {
                test_cases[a] =
                        arr(((scala.Tuple5) elt)._1(), ((scala.Tuple5) elt)
                                ._2(), ((scala.Tuple5) elt)._3(),
                                ((scala.Tuple5) elt)._4(), ((scala.Tuple5) elt)
                                        ._5());
                assertSlow(nparams == 5, "got 5 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple6) {
                test_cases[a] =
                        arr(((scala.Tuple6) elt)._1(), ((scala.Tuple6) elt)
                                ._2(), ((scala.Tuple6) elt)._3(),
                                ((scala.Tuple6) elt)._4(), ((scala.Tuple6) elt)
                                        ._5(), ((scala.Tuple6) elt)._6());
                assertSlow(nparams == 6, "got 6 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple7) {
                test_cases[a] =
                        arr(((scala.Tuple7) elt)._1(), ((scala.Tuple7) elt)
                                ._2(), ((scala.Tuple7) elt)._3(),
                                ((scala.Tuple7) elt)._4(), ((scala.Tuple7) elt)
                                        ._5(), ((scala.Tuple7) elt)._6(),
                                ((scala.Tuple7) elt)._7());
                assertSlow(nparams == 7, "got 7 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple8) {
                test_cases[a] =
                        arr(((scala.Tuple8) elt)._1(), ((scala.Tuple8) elt)
                                ._2(), ((scala.Tuple8) elt)._3(),
                                ((scala.Tuple8) elt)._4(), ((scala.Tuple8) elt)
                                        ._5(), ((scala.Tuple8) elt)._6(),
                                ((scala.Tuple8) elt)._7(), ((scala.Tuple8) elt)
                                        ._8());
                assertSlow(nparams == 8, "got 8 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple9) {
                test_cases[a] =
                        arr(((scala.Tuple9) elt)._1(), ((scala.Tuple9) elt)
                                ._2(), ((scala.Tuple9) elt)._3(),
                                ((scala.Tuple9) elt)._4(), ((scala.Tuple9) elt)
                                        ._5(), ((scala.Tuple9) elt)._6(),
                                ((scala.Tuple9) elt)._7(), ((scala.Tuple9) elt)
                                        ._8(), ((scala.Tuple9) elt)._9());
                assertSlow(nparams == 9, "got 9 parameters, expected", nparams);
            } else if (elt instanceof scala.Tuple10) {
                test_cases[a] =
                        arr(((scala.Tuple10) elt)._1(), ((scala.Tuple10) elt)
                                ._2(), ((scala.Tuple10) elt)._3(),
                                ((scala.Tuple10) elt)._4(),
                                ((scala.Tuple10) elt)._5(),
                                ((scala.Tuple10) elt)._6(),
                                ((scala.Tuple10) elt)._7(),
                                ((scala.Tuple10) elt)._8(),
                                ((scala.Tuple10) elt)._9(),
                                ((scala.Tuple10) elt)._10());
                assertSlow(nparams == 10, "got 10 parameters, expected",
                        nparams);
            } else if (elt instanceof scala.Tuple11) {
                test_cases[a] =
                        arr(((scala.Tuple11) elt)._1(), ((scala.Tuple11) elt)
                                ._2(), ((scala.Tuple11) elt)._3(),
                                ((scala.Tuple11) elt)._4(),
                                ((scala.Tuple11) elt)._5(),
                                ((scala.Tuple11) elt)._6(),
                                ((scala.Tuple11) elt)._7(),
                                ((scala.Tuple11) elt)._8(),
                                ((scala.Tuple11) elt)._9(),
                                ((scala.Tuple11) elt)._10(),
                                ((scala.Tuple11) elt)._11());
                assertSlow(nparams == 11, "got 11 parameters, expected",
                        nparams);
            } else if (elt instanceof scala.Tuple12) {
                test_cases[a] =
                        arr(((scala.Tuple12) elt)._1(), ((scala.Tuple12) elt)
                                ._2(), ((scala.Tuple12) elt)._3(),
                                ((scala.Tuple12) elt)._4(),
                                ((scala.Tuple12) elt)._5(),
                                ((scala.Tuple12) elt)._6(),
                                ((scala.Tuple12) elt)._7(),
                                ((scala.Tuple12) elt)._8(),
                                ((scala.Tuple12) elt)._9(),
                                ((scala.Tuple12) elt)._10(),
                                ((scala.Tuple12) elt)._11(),
                                ((scala.Tuple12) elt)._12());
                assertSlow(nparams == 12, "got 12 parameters, expected",
                        nparams);
            } else if (elt instanceof scala.Tuple13) {
                test_cases[a] =
                        arr(((scala.Tuple13) elt)._1(), ((scala.Tuple13) elt)
                                ._2(), ((scala.Tuple13) elt)._3(),
                                ((scala.Tuple13) elt)._4(),
                                ((scala.Tuple13) elt)._5(),
                                ((scala.Tuple13) elt)._6(),
                                ((scala.Tuple13) elt)._7(),
                                ((scala.Tuple13) elt)._8(),
                                ((scala.Tuple13) elt)._9(),
                                ((scala.Tuple13) elt)._10(),
                                ((scala.Tuple13) elt)._11(),
                                ((scala.Tuple13) elt)._12(),
                                ((scala.Tuple13) elt)._13());
                assertSlow(nparams == 13, "got 13 parameters, expected",
                        nparams);
            } else if (elt instanceof scala.Tuple14) {
                test_cases[a] =
                        arr(((scala.Tuple14) elt)._1(), ((scala.Tuple14) elt)
                                ._2(), ((scala.Tuple14) elt)._3(),
                                ((scala.Tuple14) elt)._4(),
                                ((scala.Tuple14) elt)._5(),
                                ((scala.Tuple14) elt)._6(),
                                ((scala.Tuple14) elt)._7(),
                                ((scala.Tuple14) elt)._8(),
                                ((scala.Tuple14) elt)._9(),
                                ((scala.Tuple14) elt)._10(),
                                ((scala.Tuple14) elt)._11(),
                                ((scala.Tuple14) elt)._12(),
                                ((scala.Tuple14) elt)._13(),
                                ((scala.Tuple14) elt)._14());
                assertSlow(nparams == 14, "got 14 parameters, expected",
                        nparams);
            } else if (elt instanceof scala.Tuple15) {
                test_cases[a] =
                        arr(((scala.Tuple15) elt)._1(), ((scala.Tuple15) elt)
                                ._2(), ((scala.Tuple15) elt)._3(),
                                ((scala.Tuple15) elt)._4(),
                                ((scala.Tuple15) elt)._5(),
                                ((scala.Tuple15) elt)._6(),
                                ((scala.Tuple15) elt)._7(),
                                ((scala.Tuple15) elt)._8(),
                                ((scala.Tuple15) elt)._9(),
                                ((scala.Tuple15) elt)._10(),
                                ((scala.Tuple15) elt)._11(),
                                ((scala.Tuple15) elt)._12(),
                                ((scala.Tuple15) elt)._13(),
                                ((scala.Tuple15) elt)._14(),
                                ((scala.Tuple15) elt)._15());
                assertSlow(nparams == 15, "got 15 parameters, expected",
                        nparams);
            } else {
                test_cases[a] = arr(a);
                assertSlow(nparams == 1, "got 1 parameters, expected", nparams);
            }
        }
    }

    public static Object[] arr(Object... elts) {
        return elts;
    }

    public int get_num_counterexamples() {
        return test_cases.length;
    }

    public void initialize_before_all_tests(ScCtrlConf ctrl_conf,
            ScInputConf oracle_conf)
    {
        sketch.solution_cost = 0;
        sketch.num_asserts_passed = 0;
        sketch.ctrl_conf = ctrl_conf;
        sketch.oracle_conf = oracle_conf;
    }

    public boolean run_test(int idx) {
        try {
            main_method.invoke(sketch, test_cases[idx]);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            assertFalse("run test");
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            assertFalse("run test");
        } catch (InvocationTargetException e) {
            Throwable exception = e.getTargetException();
            if (exception instanceof RuntimeException) {
                throw ((RuntimeException) exception);
            }
            print_exception("exception while executing sketch", e);
            assertFalse();
        }
        return true;
    }

    public int get_solution_cost() {
        return sketch.solution_cost;
    }

    public ScAngelicSketchBase get_sketch() {
        return sketch;
    }
}
