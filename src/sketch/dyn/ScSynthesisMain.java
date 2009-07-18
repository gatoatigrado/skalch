package sketch.dyn;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import nu.xom.Builder;
import nu.xom.Document;
import nu.xom.Elements;
import nu.xom.ParsingException;
import nu.xom.ValidityException;
import sketch.dyn.ga.ScGaSynthesis;
import sketch.dyn.inputs.ScSolvingInputConf;
import sketch.dyn.stack.ScStackSynthesis;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.ui.sourcecode.ScSourceConstruct;
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
public class ScSynthesisMain {
    protected int nthreads;
    protected ScDynamicSketch[] sketches;
    protected ScDynamicSketch ui_sketch;
    protected ScSynthesis<?> synthesis_runtime;

    /**
     * Where everything begins.
     * @param f
     *            A scala function which will yield a new sketch
     * @param cmdopts
     *            Command options
     */
    public ScSynthesisMain(scala.Function0<ScDynamicSketch> f) {
        // initialization
        BackendOptions.initialize_defaults();
        BackendOptions.initialize_annotated();
        new ScStatsMT();
        nthreads = (int) BackendOptions.synth_opts.long_("num_threads");
        ThreadLocalMT.disable_use_current_time_millis =
                BackendOptions.synth_opts.bool_("no_clock_rand");
        // initialize ssr
        sketches = new ScDynamicSketch[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = f.apply();
        }
        ui_sketch = f.apply();
        load_ui_sketch_info();
        if (BackendOptions.ga_opts.enable) {
            synthesis_runtime = new ScGaSynthesis(sketches);
        } else {
            synthesis_runtime = new ScStackSynthesis(sketches);
        }
    }

    private void load_ui_sketch_info() {
        Class<?> cls = ui_sketch.getClass();
        String info_rc = cls.getName().replace(".", File.separator) + ".info";
        URL rc = cls.getClassLoader().getResource(info_rc);
        if (rc == null) {
            DebugOut.print_mt("no info file found", info_rc);
            return;
        }
        try {
            String text = EntireFileReader.load_file(rc.openStream());
            String[] names = text.split("\\n");
            Document doc = (new Builder()).build(new File(names[0]));
            Elements srcinfo = doc.getRootElement().getChildElements();
            for (int a = 0; a < srcinfo.size(); a++) {
                ScSourceConstruct info =
                        ScSourceConstruct.from_node(srcinfo.get(a), names[1],
                                ui_sketch);
                ui_sketch.addSourceInfo(info);
            }
        } catch (IOException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ValidityException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ParsingException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        }
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
        ScUserInterface ui =
                ScUserInterfaceManager.start_ui(synthesis_runtime, ui_sketch);
        ui.set_counterexamples(inputs);
        ScStatsMT.stats_singleton.start_synthesis();
        // actual synthesize call
        synthesis_runtime.synthesize(inputs, ui);
        // stop utilities
        ScStatsMT.stats_singleton.stop_synthesis();
    }
}
