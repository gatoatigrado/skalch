package sketch.ui.modifiers;

import sketch.dyn.stack.ScLocalStackSynthesis;
import sketch.dyn.stack.ScStack;
import sketch.dyn.stats.ScStatsMT;
import sketch.util.DebugOut;

public abstract class ScUiModifierInner {
    public abstract void apply();

    /** make sure to make this method threadsafe! */
    public void setInfo(ScLocalStackSynthesis local_synth,
            ScLocalStackSynthesis.SynthesisThread synth_thread, ScStack stack)
    {
        DebugOut.assertFalse("don't append UIModifiers to objects "
                + "they don't have a setInfoInner() for");
    }

    /** make sure to make this method threadsafe! */
    public void setInfo(ScStatsMT stats) {
        DebugOut.assertFalse("don't append UIModifiers to objects "
                + "they don't have a setInfoInner() for");
    }
}
