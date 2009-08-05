package sketch.ui.modifiers;

import sketch.ui.ScUiQueueableInactive;

/**
 * print the longest stack for completed synthesis
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLongestStack extends ScLocalSynthDispatcher {
    public ScLongestStack(ScLocalSynthDispatcher prev) {
        super(prev);
    }

    @Override
    public String toString() {
        return "longest stack for stack synthesis " + local_ssr.uid;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    // the modifier...
    public class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.auto_display_first_solution = false;
            ui_thread.gui.fillWithStack(local_ssr.longest_stack);
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }
}
