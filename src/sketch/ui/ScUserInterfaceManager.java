package sketch.ui;

import sketch.dyn.BackendOptions;
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
        if (BackendOptions.ui_opts.bool_("no_gui")) {
            return new ScDebugConsoleUI(sketch);
        } else {
            ScUiThread thread = new ScUiThread(synth_runtime, sketch);
            thread.start();
            return thread;
        }
    }
}
