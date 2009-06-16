package sketch.dyn;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import sketch.dyn.inputs.ScInputConf;
import sketch.dyn.synth.ScStackSynthesis;
import sketch.util.DebugOut;

/**
 * Where everything begins.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesis {
    // FIXME - hack!
    protected int nthreads = 1;// Runtime.getRuntime().availableProcessors();
    protected ScDynamicSketch[] sketches;
    protected ScStackSynthesis ssr;

    public ScSynthesis(ScDynamicSketch sketch) throws SecurityException,
            NoSuchMethodException, IllegalArgumentException,
            InstantiationException, IllegalAccessException,
            InvocationTargetException {
        sketches = new ScDynamicSketch[nthreads];
        Constructor<?> constructor = sketch.getClass().getDeclaredConstructor();
        constructor.setAccessible(true);
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = (ScDynamicSketch) constructor.newInstance();
        }
        ssr = new ScStackSynthesis(sketches);
        DebugOut.print("ssr", ssr);
    }

    public void synthesize(ScTestGenerator tg) {
        tg.init(sketches[0].get_input_info());
        tg.tests();
        DebugOut.print((Object[]) tg.get_inputs());
        ScInputConf[] inputs = tg.get_inputs();
        ssr.synthesize(inputs);
    }
}
