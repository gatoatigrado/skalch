package sketch.dyn.main.angelic;

import java.lang.reflect.Method;

import sketch.dyn.ctrls.ScCtrlConf;
import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;

public class ScAngelicSketchCall implements
        ScDynamicSketchCall<ScAngelicSketchBase>
{
    public Object[] tuple_array;
    public final ScAngelicSketchBase sketch;
    public Method main_method;

    public ScAngelicSketchCall(ScAngelicSketchBase sketch) {
        this.sketch = sketch;
        for (Method m : sketch.getClass().getMethods()) {
            if (m.getName().equals("main")) {
                main_method = m;
            }
        }
    }

    public void set_counterexamples(Object[] tuple_array) {
        this.tuple_array = tuple_array;
    }

    public int get_num_counterexamples() {
        return tuple_array.length;
    }

    public void initialize_before_all_tests(ScCtrlConf ctrl_conf,
            ScInputConf oracle_conf)
    {
        sketch.solution_cost = 0;
        sketch.num_asserts_passed = 0;
        sketch.ctrl_conf = ctrl_conf;
        sketch.oracle_conf = oracle_conf;
    }

    public boolean run_test() {
        return true;
    }

    public void set_counterexample(int idx) {
    }

    public int get_solution_cost() {
        return sketch.solution_cost;
    }

    public ScAngelicSketchBase get_sketch() {
        return sketch;
    }
}
