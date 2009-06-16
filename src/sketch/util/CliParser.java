package sketch.util;

import java.util.LinkedList;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;

public class CliParser extends org.apache.commons.cli.PosixParser {
    public LinkedList<CliOptGroup> opt_groups = new LinkedList<CliOptGroup>();
    public CommandLine cmd_line;
    public String[] args;

    public CliParser(String[] args) {
        super();
        DebugOut.print("Cli Parser");
        DebugOut.print((Object[])args);
        this.args = args;
    }

    public void parse() {
        if (cmd_line != null) {
            return;
        }
        // add names
        Options options = new Options();
        for (CliOptGroup group : opt_groups) {
            String prefix = (group.prefixes == null) ? "" : group.prefixes[0];
            for (CliOptGroup.CmdOption cmd_opt : group.opt_set.values()) {
                options.addOption(cmd_opt.as_option(prefix));
            }
        }
        try {
            cmd_line = super.parse(options, this.args, true);
        } catch (org.apache.commons.cli.ParseException e) {
            DebugOut.assert_(false, e.getMessage());
        }
    }
}
