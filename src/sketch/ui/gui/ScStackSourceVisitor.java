package sketch.ui.gui;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import sketch.dyn.ctrls.ScCtrlSourceInfo;
import sketch.ui.sourcecode.ScSourceCache;
import sketch.util.DebugOut;

public class ScStackSourceVisitor extends ScHighlightSourceVisitor {
    @Override
    public String visitHoleInfo(ScCtrlSourceInfo ctrl_src_info) {
        if (!ctrl_src_info.src_loc.start_eq_to_end()) {
            DebugOut.assertFalse("start not equal to end for HoleInfo");
        }
        String line =
                ScSourceCache.singleton().getLine(
                        ctrl_src_info.src_loc.filename,
                        ctrl_src_info.src_loc.start.line);
        Pattern whitespace = Pattern.compile("(\\s*).*");
        Matcher m = whitespace.matcher(line);
        if (!m.matches()) {
            DebugOut.assertFalse("empty-set regex '" + whitespace.toString()
                    + "' must be able to match + '" + line + "'", m.find());
        }
        String indent = m.group(1);
        return indent + "<span style=\"color: #0000ff;\">// hole info: "
                + ctrl_src_info.info.valueString() + "</span>\n";
    }
}
