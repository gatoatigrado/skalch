package sketch.dyn.ga;

import sketch.dyn.ScClonedConstructInfo;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScGaCtrlConf;
import sketch.dyn.inputs.ScGaInputConf;
import sketch.dyn.synth.ScLocalSynthesis;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

public class ScLocalGaSynthesis extends ScLocalSynthesis {
    protected ScGaSynthesis gasynth;

    public ScLocalGaSynthesis(ScDynamicSketch sketch, ScGaSynthesis gasynth,
            int uid)
    {
        super(sketch, uid);
        this.gasynth = gasynth;
    }

    @Override
    public void run_inner() {
        thread = new SynthesisThread();
        thread.start();
    }

    public final static int NUM_BLIND_FAST = 8192;

    public class SynthesisThread extends AbstractSynthesisThread {
        protected boolean exhausted;
        protected int nruns = 0, ncounterexamples = 0;
        protected ScGaIndividual current_individual;
        protected ScGaCtrlConf ctrl_conf;
        protected ScGaInputConf oracle_conf;

        private boolean blind_fast_routine() {
            for (int a = 0; a < NUM_BLIND_FAST; a++) {
                DebugOut.not_implemented("blind_fast_routine() for ga");
            }
            return false;
        }

        @Override
        public void run_inner() {
            ScClonedConstructInfo[] info =
                    ScClonedConstructInfo.clone_array(sketch.get_hole_info());
            ctrl_conf = new ScGaCtrlConf(info);
            for (long a = 0; !gasynth.wait_handler.synthesis_complete.get(); a +=
                    NUM_BLIND_FAST)
            {
                if (gasynth.debug_stop_after != -1
                        && a >= gasynth.debug_stop_after)
                {
                    gasynth.wait_handler.wait_exhausted();
                }
                //
                // NOTE to readers: main call
                exhausted = blind_fast_routine();
                update_stats();
                gasynth.wait_handler.throw_if_synthesis_complete();
                if (!ui_queue.isEmpty()) {
                    ui_queue.remove().setInfo(ScLocalGaSynthesis.this, this,
                            null);
                }
                if (exhausted) {
                    gasynth.wait_handler.wait_exhausted();
                    gasynth.wait_handler.throw_if_synthesis_complete();
                }
            }
        }

        @Override
        public void process_ui_queue(ScUiModifier ui_modifier) {
            ui_modifier.setInfo(ScLocalGaSynthesis.this, this,
                    current_individual);
        }
    }
}
