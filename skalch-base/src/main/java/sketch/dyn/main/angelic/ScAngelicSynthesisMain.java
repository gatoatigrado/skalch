package sketch.dyn.main.angelic;

import java.util.Set;

import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.entanglement.ui.EntanglementConsole;
import sketch.result.ScSynthesisResults;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.ui.queues.QueueOutput;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.thread.AsyncMTEvent;

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
    private Set<ScSourceConstruct> sourceInfo;
    public AsyncMTEvent done_events = new AsyncMTEvent();
    private ScAngelicSketchCall queueSketch;
    private ScAngelicSketchCall entanglementSketch;
    private Set<ScSourceConstruct> entanglementSourceInfo;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        sketches = new ScAngelicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScAngelicSketchCall(f.apply());
        }
        uiSketch = new ScAngelicSketchCall(f.apply());
        queueSketch = new ScAngelicSketchCall(f.apply());
        entanglementSketch = new ScAngelicSketchCall(f.apply());

        sourceInfo = getSourceCodeInfo(uiSketch);
        entanglementSourceInfo = getSourceCodeInfo(entanglementSketch);

        synthesisRuntime = getSynthesisRuntime(sketches);

    }

    public Object synthesize() throws Exception {
        // start various utilities
        ScSynthesisResults results = new ScSynthesisResults();
        EntanglementConsole console =
                new EntanglementConsole(System.in, results, entanglementSketch,
                        entanglementSourceInfo);
        console.start();

        if (beOpts.synthOpts.queueFilename != "") {
            results.registerObserver(new QueueOutput(queueSketch,
                    beOpts.synthOpts.queueFilename));
        }

        ScUserInterface ui =
                ScUserInterfaceManager.startUi(results, beOpts, uiSketch, sourceInfo);

        initStats(ui);
        ScStatsMT.statsSingleton.startSynthesis();
        done_events.reset();
        done_events.enqueue(results, "synthesisFinished");

        // actual synthesize call
        synthesisRuntime.synthesize(results);

        // stop utilities
        done_events.set_done();
        ScStatsMT.statsSingleton.showStatsWithUi();
        return synthesisRuntime.getSolutionTuple();
    }
}
