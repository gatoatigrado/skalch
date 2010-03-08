package sketch.ui.modifiers;

import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.dyn.synth.stack.ScStack;
import sketch.util.DebugOut;

/**
 * relevant parts of the modifier which don't have to worry about the thread callbacks.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public abstract class ScUiModifierInner {
    /** always called with the Swing gui thread active */
    public abstract void apply();

    /** make sure to make this method threadsafe! */
    public void setInfo(ScLocalStackSynthesis localSynth,
            ScLocalStackSynthesis.ScStackSynthesisThread synthThread, ScStack stack)
    {
        DebugOut.assertFalse("don't append UIModifiersInner to objects "
                + "it don't have a setInfo() for");
    }

    /** make sure to make this method threadsafe! */
    public void setInfo(ScStatsMT stats) {
        DebugOut.assertFalse("don't append UIModifiersInner to objects "
                + "it don't have a setInfo() for");
    }
}
