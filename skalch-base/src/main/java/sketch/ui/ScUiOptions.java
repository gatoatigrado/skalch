package sketch.ui;

import sketch.util.cli.CliAnnotatedOptionGroup;
import sketch.util.cli.CliParameter;

/**
 * command line options for the user interface
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScUiOptions extends CliAnnotatedOptionGroup {
    public ScUiOptions() {
        super("ui", "user interface options");
    }

    @CliParameter(help = "disable the GUI (not recommended)")
    public boolean no_gui;
    @CliParameter(help = "don't automatically display the first solution")
    public boolean no_auto_soln_disp;
    @CliParameter(help = "scala file to generate after \"accept\" is clicked")
    public String accept_filename = "out.scala";
    @CliParameter(help = "number of lines surrounding the line of interest")
    public int context_len = 3;
    @CliParameter(help = "maximum amount of context between lines before they are split")
    public int context_split_len = 9;
    @CliParameter(help = "number of random stacks to save")
    public int max_random_stacks = 128;
    @CliParameter(help = "print values generated by counterexamples")
    public boolean print_counterex;
    @CliParameter(help = "regex to split lines")
    public String linesep_regex = "\\n";
    @CliParameter(help = "maximum number of counterexamples and solutions (to avoid overflowing the JList)")
    public int max_list_length = 10000;
    @CliParameter(help = "don't re-execute the program to get skdprint values")
    public boolean no_con_skdprint;
    @CliParameter(help = "don't scroll the debug out and source panes to the top left")
    public boolean no_scroll_topleft;
    @CliParameter(help = "disable bash color")
    public boolean no_bash_color;
    @CliParameter(help = "display each individual evaluated")
    public boolean display_animated;
}
