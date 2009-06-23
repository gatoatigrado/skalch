package sketch.ui.sourcecode;

import java.io.File;

import sketch.util.DebugOut;

/**
 * a location in source code. always contains a filename, sometimes line number,
 * sometimes column.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceLocation implements Comparable<ScSourceLocation> {
    public String filename;
    public LineColumn start, end;

    /**
     * creates a source location representing a 0-width caret at the beginning
     * of a given line
     */
    public ScSourceLocation(String filename, int line) {
        this.filename = filename;
        start = new LineColumn(line, 0);
        end = new LineColumn(line, 0);
    }

    public ScSourceLocation(String filename, LineColumn start, LineColumn end) {
        this.filename = filename;
        this.start = start;
        this.end = end;
    }

    @Override
    public String toString() {
        String basename = new File(filename).getName();
        return "SrcLocation[" + basename + ":" + start.toString() + " to "
                + end.toString() + "]";
    }

    public static class LineColumn {
        public int line;
        public int column;

        public LineColumn(int line, int column) {
            this.line = line;
            this.column = column;
            if (line < 0 || column < 0) {
                DebugOut.assertFalse("invalid line, column", line, column);
            }
        }

        @Override
        public String toString() {
            return line + "," + column;
        }

        public boolean lessThan(LineColumn other) {
            return (line < other.line)
                    || ((line == other.line) && (column < other.column));
        }
    }

    public int compareTo(ScSourceLocation other) {
        if (filename != other.filename) {
            return filename.compareTo(other.filename);
        } else if (start.lessThan(other.start)) {
            return -1;
        } else if (other.start.lessThan(start)) {
            return 1;
        } else {
            return 0;
        }
    }
}
