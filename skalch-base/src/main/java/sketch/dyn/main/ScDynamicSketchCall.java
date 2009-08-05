package sketch.dyn.main;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;

/**
 * provide common functionality (run, etc.) for sketches; mostly a way to
 * abstract how counterexamples are provided for now.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public interface ScDynamicSketchCall<T> {
    public int get_num_counterexamples();

    public void set_counterexample(int idx);

    public void initialize_before_all_tests(ScCtrlConf ctrl_conf,
            ScInputConf oracle_conf);

    public boolean run_test();

    public int get_solution_cost();

    public T get_sketch();
}
