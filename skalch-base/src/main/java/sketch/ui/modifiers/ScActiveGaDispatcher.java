package sketch.ui.modifiers;

import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.ga.ScLocalGaSynthesis;
import sketch.dyn.synth.ga.ScLocalGaSynthesis.ScGaSynthesisThread;
import sketch.dyn.synth.ga.base.ScGaIndividual;
import sketch.ui.ScUiList;
import sketch.ui.ScUiQueueableInactive;
import sketch.ui.gui.ScUiThread;

public class ScActiveGaDispatcher extends ScModifierDispatcher {
    protected ScGaSynthesis gasynth;

    public ScActiveGaDispatcher(ScUiThread ui_thread,
            ScUiList<ScModifierDispatcher> list, ScGaSynthesis gasynth)
    {
        super(ui_thread, list);
        this.gasynth = gasynth;
        gasynth.done_events.enqueue(this, "synthDone");
    }

    private class SynthDoneModifier extends ScUiModifierInner {
        @Override
        public void apply() {
            ui_thread.gui.num_synth_active -= 1;
            if (ui_thread.gui.num_synth_active <= 0) {
                ui_thread.gui.disableStopButton();
            }
            list.remove(ScActiveGaDispatcher.this);
        }
    }

    public void synthDone() {
        try {
            new ScUiModifier(ui_thread, new SynthDoneModifier()).enqueueTo();
        } catch (ScUiQueueableInactive e) {
            e.printStackTrace();
        }
    }

    @Override
    public void enqueue(ScUiModifier m) throws ScUiQueueableInactive {
        m.enqueueTo(gasynth);
    }

    public class Modifier extends ScUiModifierInner {
        protected ScGaIndividual individual;

        @Override
        public void apply() {
            ui_thread.gui.fillWithGaIndividual(individual);
        }

        @Override
        public void setInfo(ScLocalGaSynthesis gasynth, ScGaSynthesisThread thread,
                ScGaIndividual individual)
        {
            this.individual = individual.clone();
        }
    }

    @Override
    public ScUiModifierInner get_modifier() {
        return new Modifier();
    }

    @Override
    public String toString() {
        return "current ga synth";
    }
}
