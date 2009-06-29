package sketch.ui.sourcecode;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import sketch.util.DebugOut;

public class ScStackSourceVisitor extends ScHighlightSourceVisitor {
    @Override
    public String visitHoleInfo(ScSourceConstruct ctrl_src_info) {
        if (ctrl_src_info.entire_location.start_eq_to_end()) {
            String line =
                    ScSourceCache.singleton().getLine(
                            ctrl_src_info.entire_location.filename,
                            ctrl_src_info.entire_location.start.line);
            Pattern whitespace = Pattern.compile("(\\s*).*");
            Matcher m = whitespace.matcher(line);
            if (!m.matches()) {
                DebugOut.assertFalse("empty-set regex '"
                        + whitespace.toString() + "' must be able to match + '"
                        + line + "'", m.find());
            }
            String indent = m.group(1);
            DebugOut.todo("support non-line based holes");
            return indent + "<span style=\"color: #0000ff;\">// hole info: "
                    + ctrl_src_info.construct_info.valueString(null)
                    + "</span>\n";
        } else {
            ScSourceLocation argloc = ctrl_src_info.argument_location;
            String arg = ScSourceCache.singleton().getSourceString(argloc);
            return "<b>" + ctrl_src_info.construct_info.valueString(arg)
                    + "</b>";
        }
    }
}
