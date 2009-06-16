package sketch.util;

/**
 * A few string functions that Python has and Java doesn't.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class RichString {
    public String str;

    public RichString(String base) {
        str = base;
    }

    public String join(String[] arr) {
        if (arr == null) {
            return "<null list>";
        }
        StringBuilder b = new StringBuilder();
        for (String item : arr) {
            if (b.length() != 0) {
                b.append(str);
            }
            b.append(item);
        }
        return b.toString();
    }

    public String join(Object[] arr) {
        if (arr == null) {
            return "<null list>";
        }
        String[] as_string = new String[arr.length];
        for (int a = 0; a < arr.length; a++) {
            if (arr[a] == null) {
                as_string[a] = "<null>";
            } else {
                as_string[a] = arr[a].toString();
            }
        }
        return join(as_string);
    }

    // java's string class is kinda thin
    public String rtrim(String trim) {
        while (true) {
            int start_idx = str.length() - trim.length();
            if (start_idx >= str.length()
                    || !str.substring(start_idx).equals(trim)) {
                break;
            }
            str = str.substring(0, start_idx);
        }
        return str;
    }
}
