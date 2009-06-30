package sketch.ui.sourcecode;

import java.util.regex.Pattern;

import sketch.util.RichString;

public class ScHighlightSourceVisitor extends ScSourceLocationVisitor {
    @Override
    public String visitCode(ScSourceLocation location) {
        String[] lines = ScSourceCache.singleton().getLines(location);
        for (int a = 0; a < lines.length; a++) {
            lines[a] = highlight(lines[a]);
        }
        return (new RichString("\n")).join(lines);
    }

    private String highlight(String line) {
        line = line.replace("<", "&lt;");
        line = line.replace(">", "&gt;");
        if (Pattern.matches("^\\s*//.*", line)) {
            line = "<span style=\"color: #666666;\">" + line + "</span>";
        } else {
            line =
                    line.replaceAll("([^\\w])(val|var|def|while|if|else|for|"
                            + "return|catch|case|try|match|object|class|"
                            + "extends)([^\\w])", "$1<b>$2</b>$3");
            line =
                    line.replaceAll("(synthAssertTerminal|skprint)",
                            "<i>$1</i>");
        }
        return line;
    }
}
