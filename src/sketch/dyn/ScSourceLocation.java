package sketch.dyn;

/**
 * a location in source code. always contains a filename, sometimes line number,
 * sometimes column.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceLocation {
    public String filename;
    public LineColumn start, end;

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

    public static class LineColumn {
        public int line;
        public int column;

        public LineColumn(int line, int column) {
            this.line = line;
            this.column = column;
        }
    }
}
