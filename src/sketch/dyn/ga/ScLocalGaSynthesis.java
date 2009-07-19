package sketch.dyn.ga;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.assertFalse;

import java.util.Vector;

import sketch.dyn.ScClonedConstructInfo;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScGaCtrlConf;
import sketch.dyn.ga.analysis.GaAnalysis;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScGaInputConf;
import sketch.dyn.synth.ScDynamicUntilvException;
import sketch.dyn.synth.ScLocalSynthesis;
import sketch.dyn.synth.ScSynthesisAssertFailure;
import sketch.ui.modifiers.ScUiModifier;
import sketch.util.DebugOut;

/**
 * Container for a GA synthesis thread. The actual thread is an inner class
 * because the threads will die between synthesis rounds, whereas the sketch
 * object doesn't need to be deleted.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScLocalGaSynthesis extends ScLocalSynthesis {
    protected ScGaSynthesis gasynth;

    public ScLocalGaSynthesis(ScDynamicSketch sketch, ScGaSynthesis gasynth,
            int uid)
    {
        super(sketch, uid);
        this.gasynth = gasynth;
    }

    @Override
    protected AbstractSynthesisThread create_synth_thread() {
        DebugOut.assertSlow(gasynth.wait_handler != null, "wait_null");
        return new SynthesisThread();
    }

    public class SynthesisThread extends AbstractSynthesisThread {
        protected boolean exhausted;
        protected ScGaIndividual current_individual;
        protected ScGaCtrlConf ctrl_conf;
        protected ScGaInputConf oracle_conf;
        protected Vector<ScPopulation> local_populations;
        protected GaAnalysis analysis;

        /** evaluate $this.current_individual$ */
        protected void evaluate() {
            current_individual.set_for_synthesis_and_reset(sketch, ctrl_conf,
                    oracle_conf);
            sketch.solution_cost = 0;
            sketch.num_asserts_passed = 0;
            nruns += 1;
            trycatch: try {
                for (ScFixedInputConf counterexample : counterexamples) {
                    ncounterexamples += 1;
                    counterexample.set_input_for_sketch(sketch);
                    if (!sketch.dysketch_main()) {
                        break trycatch;
                    }
                }
                nsolutions += 1;
                gasynth.add_solution(current_individual);
            } catch (ScSynthesisAssertFailure e) {
            } catch (ScDynamicUntilvException e) {
                DebugOut.assertFalse("ga shouldn't get any "
                        + "dynamic untilv exceptions.");
            }
        }

        /** evaluate all pending individuals from all local populations */
        private void blind_fast_routine() {
            for (ScPopulation population : local_populations) {
                // print_colored(BASH_DEFAULT, "\n[ga-synth]", " ", false,
                // "population " + population.uid);
                while (population.test_queue_nonempty()) {
                    current_individual =
                            population.get_individual_for_testing();
                    // print("evaluating", "\n"
                    // + current_individual.valuesString());
                    evaluate();
                    population.add_done(current_individual.set_done(
                            sketch.solution_cost, sketch.num_asserts_passed));
                    if (analysis != null) {
                        analysis.evaluation_done(current_individual);
                    }
                    if (animated) {
                        try {
                            gasynth.ui.displayAnimated(current_individual);
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                            assertFalse("don't interrupt threads.");
                        }
                    }
                }
            }
        }

        /** generate new individuals and kill off unfit or old ones */
        protected void iterate_populations() {
            for (ScPopulation population : local_populations) {
                population.generate_new_phase(analysis);
                population.death_phase(analysis);
            }
        }

        @Override
        protected void run_inner() {
            ScClonedConstructInfo[] info =
                    ScClonedConstructInfo.clone_array(sketch.get_hole_info());
            ScClonedConstructInfo[] oracle_info =
                    ScClonedConstructInfo.clone_array(sketch.get_oracle_info());
            local_populations = new Vector<ScPopulation>();
            for (int a = 0; a < beopts().ga_opts.num_populations; a++) {
                ScPopulation population =
                        new ScPopulation(gasynth.spine_length);
                population.perturb_parameters();
                local_populations.add(population);
            }
            ctrl_conf = new ScGaCtrlConf(info);
            oracle_conf = new ScGaInputConf(oracle_info);
            if (beopts().ga_opts.analysis) {
                analysis = new GaAnalysis();
            }
            for (long a = 0; !gasynth.wait_handler.synthesis_complete.get(); a +=
                    nruns)
            {
                // print_colored(BASH_DEFAULT, "\n\n\n[ga-synth]", " ", false,
                // "=== generation start ===");
                update_stats();
                if (a >= gasynth.debug_stop_after) {
                    gasynth.wait_handler.wait_exhausted();
                }
                //
                // NOTE to readers: main call
                blind_fast_routine();
                if (!ui_queue.isEmpty()) {
                    process_ui_queue(ui_queue.remove());
                }
                iterate_populations();
            }
        }

        @Override
        protected void update_stats() {
            super.update_stats();
            if (analysis != null) {
                analysis.update_stats();
            }
        }

        @Override
        protected void process_ui_queue(ScUiModifier ui_modifier) {
            ui_modifier.setInfo(ScLocalGaSynthesis.this, this,
                    current_individual);
        }
    }
}
