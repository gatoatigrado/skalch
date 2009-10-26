package sketch.dyn.main.angelic;

import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.queues.QueueFileOutput;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;

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
		ScUserInterface ui = ScUserInterfaceManager.start_ui(be_opts,
				synthesis_runtime, ui_sketch);
		init_stats(ui);
		ScStatsMT.stats_singleton.start_synthesis();
		// actual synthesize call
		synthesis_runtime.synthesize(ui);
		// stop utilities
		ScStatsMT.stats_singleton.showStatsWithUi();
		return synthesis_runtime.get_solution_tuple();
	}
}
