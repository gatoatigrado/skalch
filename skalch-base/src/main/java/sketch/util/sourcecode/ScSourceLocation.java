package sketch.util.sourcecode;

import java.io.File;

import sketch.util.DebugOut;
import sketch.util.wrapper.XmlEltWrapper;

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
        this(filename, start, end, false);
    }

    public ScSourceLocation(String filename, LineColumn start, LineColumn end,
            boolean canBeZeroLength)
    {
        this.filename = filename;
        this.start = start;
        this.end = end;
        boolean truth =
                start.lessThan(end)
                        || (!end.lessThan(start) && canBeZeroLength);
        if (!truth) {
            DebugOut.assertFalse("constructed invalid source location", start,
                    end);
        }
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

        public static LineColumn fromXML(XmlEltWrapper pos) {
            return new LineColumn(pos.intAttr("line") - 1, pos
                    .intAttr("column") - 1);
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

    public boolean startEqToEnd() {
        return (!start.lessThan(end)) && (!end.lessThan(start));
    }

    public ScSourceLocation sourceBetween(ScSourceLocation other) {
        if (filename != other.filename) {
            DebugOut
                    .assertFalse("requesting source between two different files");
        }
        return new ScSourceLocation(filename, end, other.start, true);
    }

    public ScSourceLocation contextBefore(int nlines) {
        LineColumn beforeStart =
                new LineColumn(Math.max(0, start.line - nlines), 0);
        return new ScSourceLocation(filename, beforeStart, start, true);
    }

    public ScSourceLocation contextAfter(int nlines) {
        int lastLine =
                ScSourceCache.singleton().cachedFiles.get(filename).lines.length - 1;
        LineColumn afterEnd =
                new LineColumn(Math.min(lastLine, end.line + nlines + 1), 0);
        return new ScSourceLocation(filename, end, afterEnd, true);
    }

    public static ScSourceLocation fromXML(String filename,
            XmlEltWrapper location, boolean canBeZeroLength)
    {
        XmlEltWrapper startElt = location.XpathElt("position[@name='start']");
        XmlEltWrapper endElt = location.XpathElt("position[@name='end']");
        LineColumn startLc = LineColumn.fromXML(startElt);
        LineColumn endLc = LineColumn.fromXML(endElt);
        return new ScSourceLocation(filename, startLc, endLc,
                canBeZeroLength);
    }

    public static ScSourceLocation fromXML(String filename,
            XmlEltWrapper location)
    {
        return fromXML(filename, location, false);
    }

    public int numLines() {
        return (end.line - start.line) + 1;
    }
}
