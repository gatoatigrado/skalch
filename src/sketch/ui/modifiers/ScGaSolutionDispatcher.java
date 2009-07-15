package sketch.ui.modifiers;

import sketch.dyn.ga.base.ScGaIndividual;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

/**
 * display a solution found by the genetic algorithm.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScGaSolutionDispatcher extends ScModifierDispatcher {
    protected ScGaIndividual my_individual;

    public ScGaSolutionDispatcher(ScGaIndividual individual,
            ScUiThread ui_thread, ScUiList<ScModifierDispatcher> list)
    {
        super(ui_thread, list);
        my_individual = individual;
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo();
    }

    public class Modifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.gui.fillWithGaIndividual(my_individual);
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }

    @Override
    public String toString() {
        return "solution [cost=" + my_individual.cost + "] "
                + my_individual.solution_id_hash;
    }

    @Override
    public boolean isAcceptable() {
        return true;
    }
}
