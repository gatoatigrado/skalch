package sketch.ui;

import sketch.dyn.BackendOptions;
import sketch.dyn.synth.ScStackSynthesis;

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
        }
        return null;
    }
}
