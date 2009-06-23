package sketch.ui.modifiers;

import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.dyn.synth.ScStack;
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
    public void setInfo(ScStats stats) {
        DebugOut.assertFalse("don't append UIModifiers to objects "
                + "they don't have a setInfoInner() for");
    }
}
