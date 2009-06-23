package sketch.ui.modifiers;

import sketch.dyn.synth.ScLocalStackSynthesis;
import sketch.ui.ScUiList;
import sketch.ui.gui.ScUiThread;

/**
 * dispatcher with a local_ssr variable, and clone-style constructor.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class ScLocalSynthDispatcher extends ScModifierDispatcher {
    public ScLocalStackSynthesis local_ssr;

    public ScLocalSynthDispatcher(ScUiThread uiThread,
            ScUiList<ScModifierDispatcher> list, ScLocalStackSynthesis local_ssr)
    {
        super(uiThread, list);
        this.local_ssr = local_ssr;
    }

    public ScLocalSynthDispatcher(ScLocalSynthDispatcher prev) {
        this(prev.ui_thread, prev.list, prev.local_ssr);
    }
}
