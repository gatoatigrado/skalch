package sketch.util;

import java.util.Vector;

/**
 * extend arrays
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScArrayUtil {
    public static int[] extend_arr(int[] arr, int sz) {
        int[] next = new int[sz];
        System.arraycopy(arr, 0, next, 0, arr.length);
        return next;
    }

    public static long[] extend_arr(long[] arr, int sz) {
        long[] next = new long[sz];
        System.arraycopy(arr, 0, next, 0, arr.length);
        return next;
    }

    public static boolean[] extend_arr(boolean[] arr, int sz) {
        boolean[] next = new boolean[sz];
        System.arraycopy(arr, 0, next, 0, arr.length);
        return next;
    }

    @SuppressWarnings("unchecked")
    public static Vector<?>[] extend_arr(Vector<?>[] arr, int sz,
            boolean emptyNotNull)
    {
        Vector<?>[] next = new Vector[sz];
        for (int i = 0; i < arr.length; i++) {
            next[i] = (Vector<?>) arr[i].clone();
        }
        if (emptyNotNull) {
            for (int a = arr.length; a < sz; a++) {
                next[a] = new Vector();
            }
        }
        return next;
    }

    public static int[] extend_arr(int[] arr, int sz, int defaultv) {
        int[] next = new int[sz];
        System.arraycopy(arr, 0, next, 0, arr.length);
        for (int a = arr.length; a < sz; a++) {
            next[a] = defaultv;
        }
        return next;
    }

    public static long[] extend_arr(long[] arr, int sz, long defaultv) {
        long[] next = new long[sz];
        System.arraycopy(arr, 0, next, 0, arr.length);
        for (int a = arr.length; a < sz; a++) {
            next[a] = defaultv;
        }
        return next;
    }

    public static boolean[] extend_arr(boolean[] arr, int sz, boolean defaultv)
    {
        boolean[] next = new boolean[sz];
        System.arraycopy(arr, 0, next, 0, arr.length);
        for (int a = arr.length; a < sz; a++) {
            next[a] = defaultv;
        }
        return next;
    }
}
