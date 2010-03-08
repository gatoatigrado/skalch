package sketch.ui.modifiers;

import sketch.dyn.synth.stack.ScLocalStackSynthesis;
import sketch.ui.ScUiList;
import sketch.ui.gui.ScUiThread;

/**
 * dispatcher with a localSsr variable, and clone-style constructor.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public abstract class ScLocalSynthDispatcher extends ScModifierDispatcher {
    public ScLocalStackSynthesis localSsr;

    public ScLocalSynthDispatcher(ScUiThread uiThread,
            ScUiList<ScModifierDispatcher> list, ScLocalStackSynthesis localSsr)
    {
        super(uiThread, list);
        this.localSsr = localSsr;
    }

    public ScLocalSynthDispatcher(ScLocalSynthDispatcher prev) {
        this(prev.uiThread, prev.list, prev.localSsr);
    }
}
