package sketch.ui;

import sketch.dyn.BackendOptions;
import sketch.dyn.synth.ScStackSynthesis;

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
    public static ScUserInterface start_ui(ScStackSynthesis ssr) {
        if (!BackendOptions.ui_opts.bool_("no_gui")) {
            ScUiThread thread = new ScUiThread(ssr);
            ScUiThread.gui_list.add(thread);
            thread.start();
            return thread;
        } else {
            return new ScDebugConsoleUI();
        }
    }
}
