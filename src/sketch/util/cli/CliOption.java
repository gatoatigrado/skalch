package sketch.util.cli;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;

import sketch.util.DebugOut;

final class CliOption {
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
        return new Option(null, full_name_, has_name, help);
    }

    @Override
    public String toString() {
        return String.format("CmdOption[name=%s, type=%s, default=%s, "
                + "full_name=%s, help=%s]", name_, type_, default_, full_name_,
                help_);
    }

    public Object parse(CommandLine cmd_line) {
        if (type_.equals(Boolean.class)) {
            return cmd_line.hasOption(full_name_);
        }
        if (!cmd_line.hasOption(full_name_)) {
            if (default_ == null) {
                DebugOut.print_colored(DebugOut.BASH_RED, "", " ", false,
                        "argument", name_, "is required.\n    argument info:",
                        this);
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
