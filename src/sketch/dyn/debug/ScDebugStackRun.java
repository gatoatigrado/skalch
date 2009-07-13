package sketch.dyn.debug;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.stack.ScStack;

/**
 * static functions b/c I can't put them in ScUserInterface (java annoyance)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScDebugStackRun extends ScDebugRun {
    protected ScStack stack;

    public ScDebugStackRun(ScDynamicSketch sketch, ScStack stack,
            ScFixedInputConf[] all_counterexamples)
    {
        super(sketch, all_counterexamples);
        this.stack = stack;
    }

    @Override
    public void run_init() {
        stack.set_for_synthesis(sketch);
        stack.reset_before_run();
    }
}
