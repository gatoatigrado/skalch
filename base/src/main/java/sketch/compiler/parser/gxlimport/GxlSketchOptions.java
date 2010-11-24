package sketch.compiler.parser.gxlimport;

import sketch.compiler.main.seq.SequentialSketchOptions;
import sketch.util.cli.SketchCliParser;
import sketch.util.cuda.CudaThreadBlockDim;

public class GxlSketchOptions extends SequentialSketchOptions {
    protected GxlOptions gxlOpts;

    public GxlSketchOptions(final String[] inArgs) {
        super(inArgs);
    }

    @Override
    public void preinit() {
        // this.solverOpts.synth = SynthSolvers.ABC;
        // this.solverOpts.verif = VerifSolvers.ABC;
        // this.bndOpts.unrollAmnt = 32;
        super.preinit();
    }

    @Override
    public void parseCommandline(final SketchCliParser parser) {
        this.gxlOpts = new GxlOptions();
        this.gxlOpts.parse(parser);
        super.parseCommandline(parser);
    }

    public static GxlSketchOptions getSingleton() {
        assert SequentialSketchOptions._singleton != null : "no singleton instance";
        return (GxlSketchOptions) SequentialSketchOptions._singleton;
    }

    @Override
    public CudaThreadBlockDim getCudaBlockDim() {
        return this.gxlOpts.threadBlockDim;
    }
}
