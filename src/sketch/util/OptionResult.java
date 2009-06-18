package sketch.util;

import java.util.HashMap;

import sketch.util.CliOptGroup.CmdOption;

public class OptionResult {
    CliOptGroup options;
    CliParser parser;
    protected HashMap<String, Object> cached_results;

    public OptionResult(CliOptGroup options, CliParser parser) {
        this.options = options;
        this.parser = parser;
        DebugOut.assert_(options != null && parser != null);
        this.cached_results = new HashMap<String, Object>();
    }

    protected Object get_value(String name) {
        parser.parse();
        DebugOut.assert_(cached_results != null);
        Object result = cached_results.get(name);
        if (result == null) {
            DebugOut.assert_(options.opt_set != null);
            CmdOption opt = options.opt_set.get(name);
            DebugOut.assert_(opt != null, "invalid name", name);
            result = opt.parse(parser.cmd_line);
            cached_results.put(name, result);
        }
        return result;
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
