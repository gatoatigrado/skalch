package sketch.dyn.main;

import static sketch.util.DebugOut.not_implemented;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import nu.xom.Builder;
import nu.xom.Document;
import nu.xom.Elements;
import nu.xom.ParsingException;
import nu.xom.ValidityException;
import sketch.dyn.BackendOptions;
import sketch.dyn.stats.ScStatsMT;
import sketch.dyn.synth.ScSynthesis;
import sketch.dyn.synth.ga.ScGaSynthesis;
import sketch.dyn.synth.stack.ScStackSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.DebugOut;
import sketch.util.wrapper.EntireFileReader;

public class ScSynthesisMainBase {
    protected int nthreads;
    public BackendOptions be_opts;

    public ScSynthesisMainBase() {
        BackendOptions.initialize_defaults();
        be_opts = BackendOptions.backend_opts.get();
        be_opts.initialize_annotated();
        nthreads = be_opts.synth_opts.num_threads;
    }

    protected ScSynthesis<?> get_synthesis_runtime(
            ScDynamicSketchCall<?>[] sketches)
    {
        if (be_opts.synth_opts.solver.isGa) {
            return new ScGaSynthesis(sketches, be_opts);
        } else if (be_opts.synth_opts.solver.isStack) {
            return new ScStackSynthesis(sketches, be_opts);
        } else {
            not_implemented("ScSynthesisMainBase -- create unknown solver",
                    be_opts.synth_opts.solver);
            return null;
        }
    }

    protected void load_ui_sketch_info(ScDynamicSketchCall<?> ui_sketch_call) {
        Class<?> cls = ui_sketch_call.get_sketch().getClass();
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
                                ui_sketch_call);
                ui_sketch_call.addSourceInfo(info);
            }
        } catch (IOException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ValidityException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ParsingException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        }
    }

    public void init_stats(ScUserInterface ui) {
        new ScStatsMT(ui, be_opts);
    }
}
