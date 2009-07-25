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
    /*
     * for a in $(seq 0 1); do for i in $(seq 30 37); do echo -e
     * "\x1b[${a};${i}m${a} - ${i}\x1b[0m"; done; done
     */
    public final static String BASH_RED = "0;31";
    public final static String BASH_BLUE = "0;34";
    public final static String BASH_GREEN = "0;32";
    public final static String BASH_BROWN = "0;33";
    public final static String BASH_GREY = "1;30";
    public final static String BASH_SALMON = "1;31";
    public final static String BASH_LIGHT_BLUE = "1;34";
    /** don't use BASH_BLACK for people using black-background terminals */
    public final static String BASH_DEFAULT = "0";
    public static boolean no_bash_color = false;

    public static void print_colored(String color, String prefix, String sep,
            boolean nice_arrays, Object... text)
    {
        if (nice_arrays) {
            for (int a = 0; a < text.length; a++) {
                if (text[a] == null) {
                    text[a] = "null";
                } else if (text[a].getClass().isArray()) {
                    Object[] arr = (Object[]) text[a];
                    text[a] = (new ScRichString("\n")).join(arr);
                }
            }
        }
        if (no_bash_color) {
            System.err.println(String.format("    %s ", prefix)
                    + (new ScRichString(sep)).join(text));
        } else {
            System.err.println(String
                    .format("    \u001b[%sm%s ", color, prefix)
                    + (new ScRichString(sep)).join(text) + "\u001b[0m");
        }
    }

    public static void print(Object... text) {
        print_colored(BASH_BLUE, "[debug]", " ", false, text);
    }

    public static synchronized void print_mt(Object... text) {
        print_colored(BASH_LIGHT_BLUE, thread_indentation.get() + "[debug-"
                + Thread.currentThread().getId() + "]", " ", false, text);
    }

    public static void assertFalse(Object... description) {
        print_colored(BASH_RED, "[ASSERT FAILURE] ", " ", false, description);
        assert (false);
        throw new java.lang.IllegalStateException("please enable asserts.");
    }

    /** NOTE - don't use this in fast loops, as arrays are created */
    public static void assertSlow(boolean truth, Object... description) {
        if (!truth) {
            assertFalse(description);
        }
    }

    public static void not_implemented(Object... what) {
        Object[] what_prefixed = new Object[what.length + 1];
        what_prefixed[0] = "Not implemented -";
        System.arraycopy(what, 0, what_prefixed, 1, what.length);
        assertFalse(what_prefixed);
    }

    public static void todo(Object... what) {
        print_colored(BASH_BROWN, "[TODO] ", " ", false, what);
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

    protected static ThreadIndentation thread_indentation =
            new ThreadIndentation();

    public static void print_exception(String text, Exception e) {
        print_colored(BASH_RED, "[EXCEPTION]", "\n", false, text);
        e.printStackTrace();
    }
}