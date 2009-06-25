package sketch.util;

import java.util.HashMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;

/**
 * a group of options. subclasses are initialized by making calls in the
 * constructor; see e.g. ScSynthesisOptions for an example.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public abstract class CliOptGroup {
    protected HashMap<String, CmdOption> opt_set =
            new HashMap<String, CmdOption>();
    protected String[] prefixes;

    public void prefixes(String... prefixes) {
        this.prefixes = prefixes;
    }

    public OptionResult parse(CliParser p) {
        p.opt_groups.add(this);
        return new OptionResult(this, p);
    }

    public void add(Object... options) {
        CmdOption opt = new CmdOption();
        for (Object ent : options) {
            if (ent instanceof String) {
                String as_str = (String) ent;
                if (as_str.startsWith("--")) {
                    if (as_str.contains("num") || as_str.contains("len")) {
                        opt.type_ = Long.class;
                        if (opt.default_.equals(new Boolean(false))) {
                            opt.default_ = null;
                        }
                    }
                    if (opt.name_ != null) {
                        DebugOut.assertFalse("name already exists", opt,
                                options);
                    }
                    opt.name_ = as_str.substring(2);
                } else if (opt.help_ == null) {
                    opt.help_ = as_str;
                } else {
                    // middle argument (default value) type string, just got
                    // help, so shift args
                    opt.type_ = String.class;
                    opt.default_ = opt.help_;
                    opt.help_ = as_str;
                }
            } else if (ent instanceof Long) {
                opt.type_ = Long.class;
                opt.default_ = ent;
            } else if (ent instanceof Integer) {
                opt.type_ = Long.class;
                opt.default_ = new Long(((Integer) ent).longValue());
            } else if (ent instanceof Float) {
                opt.type_ = Float.class;
                opt.default_ = ent;
            }
        }
        if (opt.name_ == null) {
            DebugOut.assertFalse("no name given", options);
        }
        if (opt.default_ != null
                && !opt.type_.isAssignableFrom(opt.default_.getClass()))
        {
            DebugOut.assertFalse("default value", opt.default_,
                    "doesn't match type", opt.type_, "; options", options);
        }
        if (opt_set.put(opt.name_, opt) != null) {
            DebugOut.assertFalse("already contained option", opt);
        }
    }

    protected final static class CmdOption {
        public Class<?> type_ = Boolean.class;
        public Object default_ = new Boolean(false);
        public String name_ = null;
        public String full_name_ = null;
        public String help_ = null;

        public Option as_option(String prefix) {
            boolean has_name = !(type_.equals(Boolean.class));
            full_name_ = prefix.equals("") ? name_ : (prefix + "_" + name_);
            String help = help_;
            if (default_ == null) {
                help += " (REQUIRED)";
            } else if (has_name) {
                help += " [default " + default_.toString() + "]";
            }
            DebugOut.print("adding option", this);
            return new Option(null, full_name_, has_name, help);
        }

        @Override
        public String toString() {
            return String.format("CmdOption[name=%s, type=%s, default=%s, "
                    + "full_name=%s, help=%s]", name_, type_, default_,
                    full_name_, help_);
        }

        public Object parse(CommandLine cmd_line) {
            if (type_.equals(Boolean.class)) {
                return cmd_line.hasOption(full_name_);
            }
            if (!cmd_line.hasOption(full_name_)) {
                if (default_ == null) {
                    DebugOut.print_colored(DebugOut.BASH_RED, "", " ", false,
                            "argument", name_,
                            "is required.\n    argument info:", this);
                    System.exit(1); // @code standards ignore
                }
                return default_;
            }
            String v = cmd_line.getOptionValue(full_name_);
            if (type_.equals(Long.class)) {
                return Long.parseLong(v);
            } else if (type_.equals(Float.class)) {
                return Float.parseFloat(v);
            } else {
                if (!type_.equals(String.class)) {
                    DebugOut.assertFalse("can't parse type ", type_);
                }
                return v;
            }
        }
    }
}
