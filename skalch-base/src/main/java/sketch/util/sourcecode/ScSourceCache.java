package sketch.util.sourcecode;

import java.io.IOException;
import java.util.HashMap;
import java.util.Set;

import sketch.util.DebugOut;
import sketch.util.sourcecode.ScSourceLocation.LineColumn;
import sketch.util.wrapper.EntireFileReader;
import sketch.util.wrapper.ScRichString;

/**
 * reads source files and caches them as an array of lines.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceCache {
    private static ScSourceCache cache;
    public HashMap<String, SourceFile> cachedFiles =
            new HashMap<String, SourceFile>();
    public static String linesep; // set from backend options

    public static ScSourceCache singleton() {
        if (cache == null) {
            cache = new ScSourceCache();
        }
        return cache;
    }

    public void addFilenames(Set<String> filenames) {
        for (String filename : filenames) {
            if (!cachedFiles.containsKey(filename)) {
                cachedFiles.put(filename, new SourceFile(filename));
            }
        }
    }

    public static class SourceFile {
        public String[] lines;

        public SourceFile(String filename) {
            try {
                lines = EntireFileReader.load_file(filename).split(linesep);
                for (String line : lines) {
                    if (line.contains("\r")) {
                        DebugOut.print("WARNING - carriage return newline "
                                + "style, try editing --ui_linesep_regex");
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
                DebugOut.assertFalse("io exception reading file", filename);
            }
        }

        public String[] getLinesCopy(LineColumn start, LineColumn end) {
            int startLine = Math.min(start.line, lines.length - 1);
            int endLine = Math.min(end.line, lines.length - 1);
            String[] result = new String[endLine - startLine + 1];
            for (int line = startLine; line <= endLine; line++) {
                result[line - startLine] = new String(lines[line]);
            }
            try {
                // do it in reverse order if lines.length == 1 (single line)
                result[result.length - 1] =
                        result[result.length - 1].substring(0, end.column);
                result[0] = result[0].substring(start.column);
            } catch (StringIndexOutOfBoundsException e) {
                e.printStackTrace();
                DebugOut.print("strings");
                DebugOut.print((Object[]) result);
                DebugOut.print("start", start, "end", end);
                return new String[0];
            }
            return result;
        }
    }

    public String getLine(String filename, int line) {
        return cachedFiles.get(filename).lines[line];
    }

    public String[] getLines(ScSourceLocation location) {
        if (location.startEqToEnd()) {
            return new String[0];
        }
        SourceFile file = cachedFiles.get(location.filename);
        return file.getLinesCopy(location.start, location.end);
    }

    public String getSourceString(ScSourceLocation location) {
        return (new ScRichString("\n")).join(getLines(location));
    }
}
