package sketch.dyn.main.angelic;

import static sketch.dyn.BackendOptions.beopts;
import static sketch.util.DebugOut.not_implemented;

import java.lang.reflect.Method;

import sketch.dyn.ga.ScGaSynthesis;
import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stack.ScStackSynthesis;
import sketch.dyn.synth.ScSynthesis;

/**
 * where it all begins... for angelic sketches (see AngelicSketch.scala)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScAngelicSynthesisMain extends ScSynthesisMainBase {
    public final ScAngelicSketchBase ui_sketch;
    protected ScAngelicSketchCall[] sketches;
    protected ScSynthesis<?> synthesis_runtime;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        sketches = new ScAngelicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScAngelicSketchCall(f.apply());
        }
        ui_sketch = f.apply();
        not_implemented("scangelicsynthesismain");
        if (beopts().ga_opts.enable) {
            synthesis_runtime = new ScGaSynthesis(sketches);
        } else {
            synthesis_runtime = new ScStackSynthesis(sketches);
        }
    }

    public void synthesize() throws Exception {
        Method test_cases = ui_sketch.getClass().getMethod("tests");
        not_implemented("get tests from method", test_cases);
    }
}
