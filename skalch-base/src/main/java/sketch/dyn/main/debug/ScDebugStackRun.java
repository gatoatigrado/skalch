package sketch.dyn.main.debug;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.main.angelic.ScAngelicSketchBase;
import sketch.dyn.synth.stack.ScStack;

/**
 * static functions b/c I can't put them in ScUserInterface (java annoyance)
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScDebugStackRun extends ScDefaultDebugRun {
    protected ScStack stack;

    public ScDebugStackRun(ScDynamicSketchCall<ScAngelicSketchBase> sketchCall,
            ScStack stack)
    {
        super(sketchCall);
        this.stack = stack;
    }

    @Override
    public void runInit() {
        stack.resetBeforeRun();
    }

    @Override
    public ScCtrlConf getCtrlConf() {
        return stack.ctrlConf;
    }

    @Override
    public ScInputConf getOracleConf() {
        return stack.oracleConf;
    }

}
