package sketch.ui.modifiers;

import sketch.dyn.ga.ScLocalGaSynthesis;
import sketch.dyn.ga.ScLocalGaSynthesis.SynthesisThread;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.stack.ScLocalStackSynthesis;
import sketch.dyn.stack.ScStack;
import sketch.dyn.stats.ScStatsMT;
import sketch.util.DebugOut;

/**
 * relevant parts of the modifier which don't have to worry about the thread
 * callbacks.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScUiModifierInner {
    /** always called with the Swing gui thread active */
    public abstract void apply();

    /** make sure to make this method threadsafe! */
    public void setInfo(ScLocalStackSynthesis local_synth,
            ScLocalStackSynthesis.SynthesisThread synth_thread, ScStack stack)
    {
        DebugOut.assertFalse("don't append UIModifiersInner to objects "
                + "it don't have a setInfo() for");
    }

    /** make sure to make this method threadsafe! */
    public void setInfo(ScStatsMT stats) {
        DebugOut.assertFalse("don't append UIModifiersInner to objects "
                + "it don't have a setInfo() for");
    }

    public void setInfo(ScLocalGaSynthesis gasynth, SynthesisThread thread,
            ScGaIndividual individual)
    {
        DebugOut.assertFalse("don't append UIModifiersInner to objects "
                + "it don't have a setInfo() for");
    }
}
