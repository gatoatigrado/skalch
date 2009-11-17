package sketch.dyn.main.angelic;

import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.ui.queues.QueueUI;
import sketch.ui.trace.TraceUI;

/**
 * where it all begins... for angelic sketches (see AngelicSketch.scala)
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScAngelicSynthesisMain extends ScSynthesisMainBase {
    public final ScAngelicSketchCall ui_sketch;
    protected final ScAngelicSketchCall[] sketches;
    protected final ScAngelicSketchCall queue_sketch;
    protected final ScSynthesis<?> synthesis_runtime;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        sketches = new ScAngelicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScAngelicSketchCall(f.apply());
        }
        ui_sketch = new ScAngelicSketchCall(f.apply());
        queue_sketch = new ScAngelicSketchCall(f.apply());
        load_ui_sketch_info(ui_sketch);
        synthesis_runtime = get_synthesis_runtime(sketches);
    }

    public Object synthesize() throws Exception {
        // start various utilities
        ScUserInterface ui = ScUserInterfaceManager.start_ui(be_opts,
                synthesis_runtime, ui_sketch);

        if (be_opts.synth_opts.trace_file_name != "") {
            ui = new TraceUI(ui, be_opts.synth_opts.trace_file_name);
        }

        if (be_opts.synth_opts.queue_file_name != ""
                || be_opts.synth_opts.queue_input_file_name != "") {
            ui = new QueueUI(ui, queue_sketch,
                    be_opts.synth_opts.queue_file_name,
                    be_opts.synth_opts.queue_input_file_name);
        }

        init_stats(ui);
        ScStatsMT.stats_singleton.start_synthesis();
        // actual synthesize call
        synthesis_runtime.synthesize(ui);
        // stop utilities
        ScStatsMT.stats_singleton.showStatsWithUi();
        return synthesis_runtime.get_solution_tuple();
    }
}
