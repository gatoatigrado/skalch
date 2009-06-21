package sketch.util;

import java.util.concurrent.ConcurrentLinkedQueue;

import sketch.dyn.BackendOptions;

public class Profiler {
    public static ConcurrentLinkedQueue<Profiler> profilers = new ConcurrentLinkedQueue<Profiler>();
    public static ProfileMonitor monitor;

    public static ThreadLocal<Profiler> profiler = new ThreadLocal<Profiler>() {
        protected Profiler initialValue() {
            return new Profiler();
        }
    };

    public enum ProfileEvent {
        SynthesisStart, SynthesisComplete, StackNext, GetHoleValue, PostGetHoleValue
    }

    /** indexed [from][to] */
    protected int[][] transition_table;
    protected int uid;
    protected ProfileEvent current;

    public Profiler() {
        int nvalues = ProfileEvent.values().length;
        transition_table = new int[nvalues][nvalues];
        uid = profilers.size();
        profilers.add(this);
    }

    @Override
    public String toString() {
        return "Profiler[uid=" + uid + ", curr=" + current + "]";
    }

    public void set_event(ProfileEvent evt) {
        current = evt;
    }

    public static void start_monitor() {
        if (BackendOptions.stat_opts.bool_("profile_enable")) {
            monitor = new ProfileMonitor();
            monitor.start();
        }
    }

    public static void stop_monitor() {
        if (BackendOptions.stat_opts.bool_("profile_enable")) {
            monitor.stop_event.release();
        }
    }

    protected static class ProfileMonitor extends InteractiveThread {
        public ProfileMonitor() {
            super(0.1f);
        }

        @Override
        public void run_inner() {
            for (Profiler prof : profilers) {
                DebugOut.print_mt(prof);
            }
        }
    }
}
