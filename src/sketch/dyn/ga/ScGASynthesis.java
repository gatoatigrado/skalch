package sketch.dyn.ga;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.synth.ScExhaustedWaitHandler;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;

/**
 * Analogue of ScStackSynthesis
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGASynthesis extends ScSynthesis {
    protected ScLocalGASynthesis[] local_synthesis;
    protected ScExhaustedWaitHandler wait_handler;

    public ScGASynthesis(ScDynamicSketch[] sketches) {
        local_synthesis = new ScLocalGASynthesis[sketches.length];
        for (int a = 0; a < sketches.length; a++) {
            local_synthesis[a] = new ScLocalGASynthesis(sketches[a], this, a);
        }
    }

    @Override
    public boolean synthesize(ScSolvingInputConf[] counterexamples,
            ScUserInterface ui)
    {
        return false;
    }
}
