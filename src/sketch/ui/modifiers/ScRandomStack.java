package sketch.ui.modifiers;

import sketch.ui.ScUiQueueableInactive;
import ec.util.ThreadLocalMT;

/**
 * print a random stack
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScRandomStack extends ScLocalSynthDispatcher {
    static ThreadLocalMT rand = new ThreadLocalMT();

    public ScRandomStack(ScLocalSynthDispatcher prev) {
        super(prev);
    }

    @Override
    public String toString() {
        return "random stack for stack synthesis " + local_ssr.uid;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    public class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.auto_display_first_solution = false;
            int rand_idx = rand.get().nextInt(local_ssr.random_stacks.size());
            ui_thread.gui.fillWithStack(local_ssr.random_stacks.get(rand_idx));
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }
}
