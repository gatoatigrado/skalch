package sketch.ui.sourcecode;

import java.io.IOException;
import java.util.HashMap;
import java.util.Set;

import sketch.ui.sourcecode.ScSourceLocation.LineColumn;
import sketch.util.DebugOut;
import sketch.util.EntireFileReader;
import sketch.util.RichString;

public class ScSourceCache {
    private static ScSourceCache cache;
    public HashMap<String, SourceFile> cached_files =
            new HashMap<String, SourceFile>();

    public static ScSourceCache singleton() {
        if (cache == null) {
            cache = new ScSourceCache();
        }
        return cache;
    }

    public void add_filenames(Set<String> filenames) {
        for (String filename : filenames) {
            if (!cached_files.containsKey(filename)) {
                cached_files.put(filename, new SourceFile(filename));
            }
        }
    }

    public static class SourceFile {
        public String[] lines;

        public SourceFile(String filename) {
            try {
                lines = EntireFileReader.load_file(filename).split("\\n");
            } catch (IOException e) {
                e.printStackTrace();
                DebugOut.assertFalse("io exception reading file", filename);
            }
        }

        public String[] getLinesCopy(LineColumn start, LineColumn end) {
            int start_line = Math.min(start.line, lines.length - 1);
            int end_line = Math.min(end.line, lines.length - 1);
            String[] result = new String[end_line - start_line + 1];
            for (int line = start_line; line <= end_line; line++) {
                result[line - start_line] = new String(lines[line]);
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
        return cached_files.get(filename).lines[line];
    }

    public String[] getLines(ScSourceLocation location) {
        if (location.start_eq_to_end()) {
            return new String[0];
        }
        SourceFile file = cached_files.get(location.filename);
        return file.getLinesCopy(location.start, location.end);
    }

    public String getSourceString(ScSourceLocation location) {
        return (new RichString("\n")).join(getLines(location));
    }
}
