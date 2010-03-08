package sketch.dyn.main.angelic;

import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.result.ScSynthesisResults;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.ui.sourcecode.ScSourceConstruct;

/**
 * where it all begins... for angelic sketches (see AngelicSketch.scala)
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScAngelicSynthesisMain extends ScSynthesisMainBase {
    public final ScAngelicSketchCall uiSketch;
    protected final ScAngelicSketchCall[] sketches;
    protected final ScSynthesis<?> synthesisRuntime;
    private ScSourceConstruct sourceInfo;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        sketches = new ScAngelicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScAngelicSketchCall(f.apply());
        }
        uiSketch = new ScAngelicSketchCall(f.apply());
        sourceInfo = getSourceCodeInfo(uiSketch);
        uiSketch.addSourceInfo(sourceInfo);
        synthesisRuntime = getSynthesisRuntime(sketches);
    }

    public Object synthesize() throws Exception {
        // start various utilities
        ScUserInterface ui =
                ScUserInterfaceManager.startUi(beOpts, synthesisRuntime, uiSketch,
                        sourceInfo);

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

        initStats(ui);
        ScStatsMT.statsSingleton.startSynthesis();
        // actual synthesize call
        synthesisRuntime.synthesize(results);
        // stop utilities
        ScStatsMT.statsSingleton.showStatsWithUi();
        return synthesisRuntime.getSolutionTuple();
    }
}
