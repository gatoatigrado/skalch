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
        if (options == null || parser == null) {
            DebugOut.assertFalse();
        }
        this.cached_results = new HashMap<String, Object>();
    }

    protected Object get_value(String name) {
        parser.parse();
        if (cached_results == null) {
            DebugOut.assertFalse();
        }
        Object result = cached_results.get(name);
        if (result == null) {
            if (options.opt_set == null) {
                DebugOut.assertFalse();
            }
            CmdOption opt = options.opt_set.get(name);
            if (opt == null) {
                DebugOut.assertFalse("invalid name", name);
            }
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
