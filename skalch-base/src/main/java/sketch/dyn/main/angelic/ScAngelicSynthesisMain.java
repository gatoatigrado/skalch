package sketch.dyn.main.angelic;

import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.result.ScSynthesisResults;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;

/**
 * where it all begins... for angelic sketches (see AngelicSketch.scala)
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScAngelicSynthesisMain extends ScSynthesisMainBase {
    public final ScAngelicSketchCall ui_sketch;
    protected final ScAngelicSketchCall[] sketches;
    protected final ScSynthesis<?> synthesis_runtime;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        sketches = new ScAngelicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScAngelicSketchCall(f.apply());
        }
        ui_sketch = new ScAngelicSketchCall(f.apply());
        load_ui_sketch_info(ui_sketch);
        synthesis_runtime = get_synthesis_runtime(sketches);
    }

    public Object synthesize() throws Exception {
        // start various utilities
        ScUserInterface ui =
                ScUserInterfaceManager.start_ui(be_opts, synthesis_runtime, ui_sketch);

        // if (be_opts.synth_opts.trace_filename != "") {
        // ui = new TraceUI(ui, be_opts.synth_opts.trace_filename);
        // }
        //
        // if (be_opts.synth_opts.entanglement) {
        // ui = new RecordTraceUI(ui);
        // }
        //
        // if (be_opts.synth_opts.queue_filename != ""
        // || be_opts.synth_opts.queuein_filename != "") {
        // ui = new QueueUI(ui, queue_sketch,
        // be_opts.synth_opts.queue_filename,
        // be_opts.synth_opts.queuein_filename);
        // }

        ScSynthesisResults results = new ScSynthesisResults();

        init_stats(ui);
        ScStatsMT.stats_singleton.start_synthesis();
        // actual synthesize call
        synthesis_runtime.synthesize(ui);
        // stop utilities
        ScStatsMT.stats_singleton.showStatsWithUi();
        return synthesis_runtime.get_solution_tuple();
    }
}
