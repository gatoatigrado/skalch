package sketch.compiler.parser.gxlimport;

import sketch.compiler.main.seq.SequentialSketchOptions;
import sketch.util.cli.SketchCliParser;

public class GxlSketchOptions extends SequentialSketchOptions {
    protected GxlOptions gxlOpts;

    public GxlSketchOptions(final String[] inArgs) {
        super(inArgs);
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
}
