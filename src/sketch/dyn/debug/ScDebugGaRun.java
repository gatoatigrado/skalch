package sketch.dyn.debug;

import sketch.dyn.ScDynamicSketch;
import sketch.dyn.ctrls.ScGaCtrlConf;
import sketch.dyn.ga.base.ScGaIndividual;
import sketch.dyn.inputs.ScFixedInputConf;
import sketch.dyn.inputs.ScGaInputConf;

public class ScDebugGaRun extends ScDebugRun {
    protected ScGaIndividual individual;
    protected ScGaCtrlConf ctrl_conf;
    protected ScGaInputConf oracle_conf;

    public ScDebugGaRun(ScDynamicSketch sketch,
            ScFixedInputConf[] all_counterexamples, ScGaIndividual individual,
            ScGaCtrlConf ctrl_conf, ScGaInputConf oracle_conf)
    {
        super(sketch, all_counterexamples);
        this.individual = individual;
        this.ctrl_conf = ctrl_conf;
        this.oracle_conf = oracle_conf;
    }

    @Override
    public void run_init() {
        individual.reset_fitness();
        individual.set_for_synthesis_and_reset(sketch, ctrl_conf, oracle_conf);
    }
}
