package sketch.dyn;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import nu.xom.Builder;
import nu.xom.Document;
import nu.xom.Element;
import nu.xom.ParsingException;
import nu.xom.ValidityException;
import scala.Function0;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stats.ScStats;
import sketch.dyn.synth.ScStackSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.util.DebugOut;
import sketch.util.EntireFileReader;
import ec.util.ThreadLocalMT;

/**
 * Where everything begins.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesis {
    protected int nthreads;
    protected ScDynamicSketch[] sketches;
    protected ScDynamicSketch ui_sketch;
    protected ScStackSynthesis ssr;

    /**
     * Where everything begins.
     * @param f
     *            A scala function which will yield a new sketch
     * @param cmdopts
     *            Command options
     */
    public ScSynthesis(scala.Function0<ScDynamicSketch> f) {
        // initialization
        BackendOptions.initialize_defaults();
        ScStats.initialize();
        nthreads = (int) BackendOptions.synth_opts.long_("num_threads");
        ThreadLocalMT.disable_use_current_time_millis =
                BackendOptions.synth_opts.bool_("no_clock_rand");
        // initialize ssr
        sketches = new ScDynamicSketch[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = load_sketch(f);
        }
        ui_sketch = load_sketch(f);
        ssr = new ScStackSynthesis(sketches);
    }

    private ScDynamicSketch load_sketch(Function0<ScDynamicSketch> f) {
        ScDynamicSketch sketch = f.apply();
        Class<?> cls = sketch.getClass();
        String info_rc = cls.getName().replace(".", File.separator) + ".info";
        URL rc = cls.getClassLoader().getResource(info_rc);
        try {
            String text = EntireFileReader.load_file(rc.openStream());
            String[] names = text.split("\\n");
            Builder b = new Builder();
            Document doc = b.build(new File(names[0]));
            Element elt = doc.getRootElement();
            DebugOut.print("root elt", elt);
        } catch (IOException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ValidityException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ParsingException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        }
        System.exit(0);
        return sketch;
    }

    protected ScSolvingInputConf[] generate_inputs(ScDynamicSketch sketch) {
        ScTestGenerator tg = sketch.test_generator();
        tg.init(ui_sketch.get_input_info());
        tg.tests();
        return tg.get_inputs();
    }

    public void synthesize() {
        ScSolvingInputConf[] inputs = generate_inputs(ui_sketch);
        // start various utilities
        ScUserInterface ui = ScUserInterfaceManager.start_ui(ssr, ui_sketch);
        ui.set_counterexamples(inputs);
        ScStats.stats.start_synthesis();
        // actual synthesize call
        ssr.synthesize(inputs, ui);
        // stop utilities
        ScStats.stats.stop_synthesis();
    }
}
