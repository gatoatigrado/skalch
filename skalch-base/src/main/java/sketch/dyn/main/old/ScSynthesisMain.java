package sketch.dyn.main.old;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import nu.xom.Builder;
import nu.xom.Document;
import nu.xom.Elements;
import nu.xom.ParsingException;
import nu.xom.ValidityException;
import sketch.dyn.constructs.inputs.ScFixedInputConf;
import sketch.dyn.constructs.inputs.ScSolvingInputConf;
import sketch.dyn.main.ScSynthesisMainBase;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.ScUserInterfaceManager;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.DebugOut;
import sketch.util.EntireFileReader;

/**
 * Where everything begins.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSynthesisMain extends ScSynthesisMainBase {
    protected final ScOldDynamicSketchCall[] sketches;
    protected final ScOldDynamicSketchCall ui_sketch;
    protected final ScSynthesis<?> synthesis_runtime;

    /**
     * Where everything begins.
     * @param f
     *            A scala function which will yield a new sketch
     * @param cmdopts
     *            Command options
     */
    public ScSynthesisMain(scala.Function0<ScOldDynamicSketch> f) {
        sketches = new ScOldDynamicSketchCall[nthreads];
        for (int a = 0; a < nthreads; a++) {
            sketches[a] = new ScOldDynamicSketchCall(f.apply());
        }
        ui_sketch = new ScOldDynamicSketchCall(f.apply());
        load_ui_sketch_info();
        synthesis_runtime = get_synthesis_runtime(sketches);
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
                                ui_sketch.get_sketch());
                ui_sketch.get_sketch().addSourceInfo(info);
            }
        } catch (IOException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ValidityException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ParsingException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        }
    }

    protected ScSolvingInputConf[] generate_inputs(ScOldDynamicSketch sketch) {
        ScTestGenerator tg = sketch.test_generator();
        tg.init(ui_sketch.get_sketch().get_input_info());
        tg.tests();
        return tg.get_inputs();
    }

    public Object synthesize() {
        ScSolvingInputConf[] inputs = generate_inputs(ui_sketch.get_sketch());
        // start various utilities
        ScUserInterface ui =
                ScUserInterfaceManager.start_ui(synthesis_runtime, ui_sketch);
        ui.set_counterexamples(inputs);
        ui_sketch.counterexamples = ScFixedInputConf.from_inputs(inputs);
        for (ScOldDynamicSketchCall sketch : sketches) {
            sketch.counterexamples = ScFixedInputConf.from_inputs(inputs);
        }
        ScStatsMT.stats_singleton.start_synthesis();
        // actual synthesize call
        synthesis_runtime.synthesize(ui);
        // stop utilities
        ScStatsMT.stats_singleton.stop_synthesis();
        return synthesis_runtime.get_solution_tuple();
    }
}
