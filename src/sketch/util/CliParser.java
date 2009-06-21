package sketch.util;

import java.util.LinkedList;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;

public class CliParser extends org.apache.commons.cli.PosixParser {
    public LinkedList<CliOptGroup> opt_groups = new LinkedList<CliOptGroup>();
    public CommandLine cmd_line;
    public String[] args;

    public CliParser(String[] args) {
        super();
        this.args = args;
    }

    public void parse() {
        if (cmd_line != null) {
            return;
        }
        // add names
        Options options = new Options();
        options.addOption("h", "help", false, "display help");
        for (CliOptGroup group : opt_groups) {
            String prefix = (group.prefixes == null) ? "" : group.prefixes[0];
            for (CliOptGroup.CmdOption cmd_opt : group.opt_set.values()) {
                options.addOption(cmd_opt.as_option(prefix));
            }
        }
        try {
            cmd_line = super.parse(options, this.args, true);
            boolean print_help = cmd_line.hasOption("help");
            for (String arg : cmd_line.getArgs()) {
                if (arg.equals("--")) {
                    break;
                } else if (arg.startsWith("--")) {
                    DebugOut.print("unrecognized argument", arg);
                    print_help = true;
                }
            }
            if (print_help) {
                HelpFormatter hf = new HelpFormatter();
                hf.printHelp("[options]", options);
                System.exit(1); // @code standards ignore
            }
        } catch (org.apache.commons.cli.ParseException e) {
            DebugOut.assert_(false, e.getMessage());
        }
    }
}
