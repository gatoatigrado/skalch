package sketch.dyn.main.old;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;

public class ScOldDynamicSketchCall implements
        ScDynamicSketchCall<ScOldDynamicSketch>
{
    public final ScOldDynamicSketch sketch;
    public ScFixedInputConf counterexamples[];

    public ScOldDynamicSketchCall(ScOldDynamicSketch sketch) {
        this.sketch = sketch;
    }

    public int get_num_counterexamples() {
        return counterexamples.length;
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
        counterexamples[idx].set_input_for_sketch(sketch);
        return sketch.dysketch_main();
    }

    public int get_solution_cost() {
        return sketch.solution_cost;
    }

    public ScOldDynamicSketch get_sketch() {
        return sketch;
    }
}
