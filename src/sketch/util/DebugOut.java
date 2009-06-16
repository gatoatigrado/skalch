package sketch.util;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Debugging utilities, including an "assert" that doesn't require -ea.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class DebugOut {
    public final static String BASH_RED = "0;31";
    public final static String BASH_BLUE = "0;34";
    public final static String BASH_GREEN = "0;32";
    public final static String BASH_ORANGE = "0;33";
    public final static String BASH_LIGHT_BLUE = "1;34";
    /** don't use BASH_BLACK for people using black-background terminals */
    public final static String BASH_DEFAULT = "0";

    public static void print_colored(String color, String prefix, String sep,
            Object[] text) {
        System.err.println(String.format("    \u001b[%sm%s ", color, prefix)
                + (new RichString(sep)).join(text) + "\u001b[0m");
    }

    public static void print(Object... text) {
        print_colored(BASH_BLUE, "[debug]", " ", text);
    }

    public static synchronized void print_mt(Object... text) {
        print_colored(BASH_LIGHT_BLUE, thread_indentation.get() + "[debug-"
                + Thread.currentThread().getId() + "]", " ", text);
    }

    public static void assert_(boolean truth, Object... description) {
        if (!truth) {
            print_colored(BASH_RED, "[ASSERT FAILURE] ", " ", description);
            assert (false);
            throw new java.lang.IllegalStateException("please enable asserts.");
        }
    }

    public static void not_implemented(Object... what) {
        Object[] what_prefixed = new Object[what.length + 1];
        what_prefixed[0] = "Not implemented -";
        System.arraycopy(what, 0, what_prefixed, 1, what.length);
        assert_(false, what_prefixed);
    }

    public static void todo(Object... what) {
        print_colored(BASH_ORANGE, "[TODO] ", " ", what);
    }

    protected static class ThreadIndentation extends ThreadLocal<String> {
        private static AtomicInteger ctr = new AtomicInteger(0);

        @Override
        protected String initialValue() {
            String result = "";
            int n_spaces = ctr.getAndIncrement();
            for (int a = 0; a < n_spaces; a++) {
                result += " ";
            }
            return result;
        }
    }

    protected static ThreadIndentation thread_indentation = new ThreadIndentation();
}
