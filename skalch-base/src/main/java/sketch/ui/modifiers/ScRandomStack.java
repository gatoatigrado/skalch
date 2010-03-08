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
        return "random stack for stack synthesis " + localSsr.uid;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    private class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            uiThread.autoDisplayFirstSolution = false;
            int randIdx = rand.get().nextInt(localSsr.randomStacks.size());
            uiThread.gui.fillWithStack(ScRandomStack.this,
                    localSsr.randomStacks.get(randIdx));
        }
    }

    @Override
    public ScUiModifierInner getModifier() {
        return new Modifier();
    }
}
