package sketch.ui.sourcecode;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import sketch.util.DebugOut;

public class ScStackSourceVisitor extends ScHighlightSourceVisitor {
    @Override
    public String visitHoleInfo(ScSourceConstruct ctrl_src_info) {
        ScSourceLocation argloc = ctrl_src_info.argument_location;
        String arg = ScSourceCache.singleton().getSourceString(argloc);
        if (ctrl_src_info.entire_location.start_eq_to_end()) {
            String line =
                    ScSourceCache.singleton().getLine(
                            ctrl_src_info.entire_location.filename,
                            ctrl_src_info.entire_location.start.line);
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
                    + ctrl_src_info.getName() + ": "
                    + ctrl_src_info.construct_info.valueString(arg)
                    + "</span>\n";
        } else {
            return "<b>" + ctrl_src_info.construct_info.valueString(arg)
                    + "</b>";
        }
    }
}
