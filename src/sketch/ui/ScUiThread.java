package sketch.ui;

import java.util.NoSuchElementException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import sketch.dyn.BackendOptions;
import sketch.dyn.synth.ScStackSynthesis;
import sketch.util.DebugOut;
import sketch.util.InteractiveThread;

/**
 * Thread which launches the user interface and shuts it down when requested.
 * Perhaps slightly unnecessary but nice for code organization.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScUiThread extends InteractiveThread implements ScUserInterface {
    protected ScStackSynthesis ssr;
    protected gui_0_1 gui;
    public AtomicInteger modifier_timestamp = new AtomicInteger(0);
    static ConcurrentLinkedQueue<ScUiThread> gui_list = new ConcurrentLinkedQueue<ScUiThread>();

    public ScUiThread(ScStackSynthesis ssr) {
        super(0.1f);
        this.ssr = ssr;
    }

    @Override
    public void init() {
        gui = new gui_0_1();
        gui.setVisible(true);
    }

    @Override
    public void run_inner() {
    }

    @Override
    public void finish() {
        gui.setVisible(false);
        gui.dispose();
    }

    public static void start_ui(ScStackSynthesis ssr) {
        if (!BackendOptions.ui_opts.bool_("no_gui")) {
            ScUiThread thread = new ScUiThread(ssr);
            gui_list.add(thread);
            thread.start();
        }
    }

    /** @deprecated deleteme soon, just use set_synthesis_complete */
    public static void stop_ui() {
        try {
            while (true) {
                ScUiThread thread = gui_list.remove();
                thread.set_stop();
            }
        } catch (NoSuchElementException e) {
        }
    }

    public void modifierComplete(ScUiModifier m) {
        DebugOut.not_implemented("modifierComplete");
    }

    public int nextModifierTimestamp() {
        return modifier_timestamp.incrementAndGet();
    }
}
