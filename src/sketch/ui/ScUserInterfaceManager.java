package sketch.ui;

import static sketch.dyn.BackendOptions.beopts;
import sketch.dyn.ScDynamicSketch;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.gui.ScUiThread;

/**
 * static functions for all user interfaces.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUserInterfaceManager {
    /**
     * start a user interface
     * @param ssr
     *            This should be removed sometime, using calls to
     *            ScUserInterface from appropriate solvers.
     */
    public static ScUserInterface start_ui(ScSynthesis<?> synth_runtime,
            ScDynamicSketch sketch)
    {
        if (beopts().ui_opts.no_gui) {
            return new ScDebugConsoleUI(sketch);
        } else {
            ScUiThread thread = new ScUiThread(synth_runtime, sketch, beopts());
            thread.start();
            return thread;
        }
    }
}
