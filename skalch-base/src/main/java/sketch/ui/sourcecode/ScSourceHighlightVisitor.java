package sketch.ui.sourcecode;

import java.util.regex.Pattern;

import sketch.util.fcns.ScHtmlUtil;
import sketch.util.sourcecode.ScSourceCache;
import sketch.util.sourcecode.ScSourceLocation;
import sketch.util.wrapper.ScRichString;

/**
 * basic scala source code highlighting (bold keywords)
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceHighlightVisitor extends ScSourceLocationVisitor {
    @Override
    public String visitCode(ScSourceLocation location) {
        String[] lines = ScSourceCache.singleton().getLines(location);
        for (int a = 0; a < lines.length; a++) {
            lines[a] = highlight(lines[a]);
        }
        return (new ScRichString("\n")).join(lines);
    }

    private String highlight(String line) {
        line = ScHtmlUtil.html_tag_escape(line);
        if (Pattern.matches("^\\s*//.*", line)) {
            line = "<span style=\"color: #666666;\">" + line + "</span>";
        } else {
            line =
                    line.replaceAll("([^\\w])(val|var|def|while|if|else|for|"
                            + "return|catch|case|try|match|object|class|"
                            + "extends|import|package)([^\\w])",
                            "$1<b>$2</b>$3");
            line =
                    line.replaceAll("(synthAssert|skprint)",
                            "<i>$1</i>");
        }
        return line;
    }
}
