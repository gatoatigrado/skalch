package sketch.util;

public class OptionResult {
    CliOptGroup options;
    CliParser parser;

    public OptionResult(CliOptGroup options, CliParser parser) {
        this.options = options;
        this.parser = parser;
    }

    protected Object get_value(String name) {
        parser.parse();
        return options.opt_set.get(name).parse(parser.cmd_line);
    }

    public boolean bool_(String name) {
        return (Boolean) get_value(name);
    }

    public String str_(String name) {
        return (String) get_value(name);
    }

    public int int_(String name) {
        return (Integer) get_value(name);
    }

    public float flt_(String name) {
        return (Float) get_value(name);
    }
}
