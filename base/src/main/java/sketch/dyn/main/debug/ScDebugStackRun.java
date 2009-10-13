package sketch.dyn.main.debug;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.synth.stack.ScStack;

/**
 * static functions b/c I can't put them in ScUserInterface (java annoyance)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScDebugStackRun extends ScDefaultDebugRun {
    protected ScStack stack;

    public ScDebugStackRun(
            ScDynamicSketchCall<ScAngelicSketchBase> sketch_call, ScStack stack)
    {
        super(sketch_call);
        this.stack = stack;
    }

    @Override
    public void run_init() {
        stack.reset_before_run();
    }

    @Override
    public ScCtrlConf get_ctrl_conf() {
        return stack.ctrl_conf;
    }

    @Override
    public ScInputConf get_oracle_conf() {
        return stack.oracle_conf;
    }
}
