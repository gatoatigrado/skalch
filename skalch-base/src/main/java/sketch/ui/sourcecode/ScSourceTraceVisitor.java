package sketch.ui.sourcecode;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import sketch.util.DebugOut;
import sketch.util.sourcecode.ScSourceCache;
import sketch.util.sourcecode.ScSourceLocation;

/**
 * formats values chosen by the synthesizer.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceTraceVisitor extends ScSourceHighlightVisitor {
    @Override
    public String visitHoleInfo(ScSourceConstruct ctrlSrcInfo) {
        ScSourceLocation argloc = ctrlSrcInfo.argumentLocation;
        String arg = ScSourceCache.singleton().getSourceString(argloc);
        if (ctrlSrcInfo.entireLocation.startEqToEnd()) {
            String line =
                    ScSourceCache.singleton().getLine(
                            ctrlSrcInfo.entireLocation.filename,
                            ctrlSrcInfo.entireLocation.start.line);
            Pattern whitespace = Pattern.compile("(\\s*).*");
            Matcher m = whitespace.matcher(line);
            String indent = "";
            if (!m.matches()) {
                DebugOut.print("empty-set regex '" + whitespace.toString()
                        + "' must be able to match + '" + line + "'", m.find());
                indent = m.group(1);
            }
            return indent
                    + "<span style=\"color: #0000ff;\">// construct info for "
                    + ctrlSrcInfo.getName() + ": "
                    + ctrlSrcInfo.constructInfo.valueString(arg)
                    + "</span>\n";
        } else {
            return "<b>" + ctrlSrcInfo.constructInfo.valueString(arg)
                    + "</b>";
        }
    }
}
