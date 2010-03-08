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
    public static ScUserInterface startUi(BackendOptions beOpts,
            ScSynthesis<?> synthRuntime, ScDynamicSketchCall<?> sketch,
            ScSourceConstruct sourceInfo)
    {
        if (beOpts.uiOpts.noGui) {
            return new ScDebugConsoleUI(beOpts, sketch);
        } else {
            ScUiThread thread =
                    new ScUiThread(synthRuntime, sketch, beOpts, sourceInfo);
            thread.start();
            return thread;
        }
    }
}
