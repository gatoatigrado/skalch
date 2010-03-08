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
import sketch.dyn.synth.stack.ScStackSynthesis;
import sketch.ui.ScUserInterface;
import sketch.ui.sourcecode.ScSourceConstruct;
import sketch.util.DebugOut;
import sketch.util.wrapper.EntireFileReader;

public class ScSynthesisMainBase {
    protected int nthreads;
    public BackendOptions beOpts;

    public ScSynthesisMainBase() {
        BackendOptions.initializeDefaults();
        beOpts = BackendOptions.backendOpts.get();
        beOpts.initializeAnnotated();
        nthreads = beOpts.synthOpts.numThreads;
    }

    protected ScSynthesis<?> getSynthesisRuntime(ScDynamicSketchCall<?>[] sketches) {
        if (beOpts.synthOpts.solver.isStack) {
            return new ScStackSynthesis(sketches, beOpts);
        } else {
            not_implemented("ScSynthesisMainBase -- create unknown solver",
                    beOpts.synthOpts.solver);
            return null;
        }
    }

    protected ScSourceConstruct getSourceCodeInfo(ScDynamicSketchCall<?> uiSketchCall) {
        Class<?> cls = uiSketchCall.getSketch().getClass();
        String infoRc = cls.getName().replace(".", File.separator) + ".info";
        URL rc = cls.getClassLoader().getResource(infoRc);
        if (rc == null) {
            DebugOut.print_mt("No source info file found.", infoRc);
            return null;
        }
        try {
            String text = EntireFileReader.load_file(rc.openStream());
            String[] names = text.split("\\n");
            Document doc = (new Builder()).build(new File(names[0]));
            Elements srcinfo = doc.getRootElement().getChildElements();
            for (int a = 0; a < srcinfo.size(); a++) {
                ScSourceConstruct info =
                        ScSourceConstruct.fromNode(srcinfo.get(a), names[1],
                                uiSketchCall);
                return info;
            }
        } catch (IOException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ValidityException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        } catch (ParsingException e) {
            DebugOut.print_exception("reading source annotation info ", e);
        }
        DebugOut.print_mt("Exception while reading source info file.", infoRc);
        return null;
    }

    public void initStats(ScUserInterface ui) {
        new ScStatsMT(ui, beOpts);
    }
}
