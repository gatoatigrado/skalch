package sketch.dyn.main;

import static sketch.dyn.BackendOptions.beopts;
import sketch.dyn.BackendOptions;

public class ScAngelicSynthesisMain {
    public final ScAngelicSketchBase ui_sketch;

    public ScAngelicSynthesisMain(scala.Function0<ScAngelicSketchBase> f) {
        BackendOptions.initialize_defaults();
        beopts().initialize_annotated();
        ui_sketch = f.apply();
    }

    public void synthesize() {
    }
}
