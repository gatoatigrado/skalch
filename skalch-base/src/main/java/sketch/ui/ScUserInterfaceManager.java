package sketch.ui;

import sketch.dyn.BackendOptions;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.ScSynthesis;
import sketch.ui.gui.ScUiThread;
import sketch.ui.sourcecode.ScSourceConstruct;

/**
 * static functions for all user interfaces.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScUserInterfaceManager {
    /**
     * start a user interface
     * 
     * @param sourceInfo
     * @param ssr
     *            This should be removed sometime, using calls to ScUserInterface from
     *            appropriate solvers.
     */
    public static ScUserInterface start_ui(BackendOptions be_opts,
            ScSynthesis<?> synth_runtime, ScDynamicSketchCall<?> sketch,
            ScSourceConstruct sourceInfo)
    {
        if (be_opts.ui_opts.no_gui) {
            return new ScDebugConsoleUI(be_opts, sketch);
        } else {
            ScUiThread thread =
                    new ScUiThread(synth_runtime, sketch, be_opts, sourceInfo);
            thread.start();
            return thread;
        }
    }
}
