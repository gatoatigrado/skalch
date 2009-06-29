package sketch.ui.sourcecode;

import nu.xom.Element;
import sketch.dyn.ScDynamicSketch;
import sketch.ui.sourcecode.ScSourceLocation.LineColumn;
import sketch.util.DebugOut;
import sketch.util.XmlEltWrapper;

/**
 * A source construct info bound to a location in source.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceConstruct implements Comparable<ScSourceConstruct> {
    public ScSourceConstructInfo construct_info;
    public ScSourceLocation entire_location;
    public ScSourceLocation argument_location;

    public ScSourceConstruct(ScSourceConstructInfo construct_info,
            ScSourceLocation entire_location, ScSourceLocation argument_location)
    {
        this.construct_info = construct_info;
        this.entire_location = entire_location;
        this.argument_location = argument_location;
    }

    @Override
    public String toString() {
        return "ScSourceConstruct[entire_loc=" + entire_location.toString()
                + ", arg_loc=" + argument_location.toString() + "]";
    }

    public String getName() {
        return construct_info.getName() + "@"
                + entire_location.start.toString();
    }

    public static ScSourceConstruct from_node(Element child_, String filename,
            ScDynamicSketch sketch)
    {
        XmlEltWrapper elt = new XmlEltWrapper(child_);
        int uid = elt.int_attr("uid");
        XmlEltWrapper entire_loc = elt.XpathElt("rangepos[@name='entire_pos']");
        XmlEltWrapper argument_loc = elt.XpathElt("rangepos[@name='arg_pos']");
        ScSourceConstructInfo cons_info = null;
        ScSourceLocation eloc = ScSourceLocation.fromXML(filename, entire_loc);
        if (elt.getLocalName().equals("holeapply")) {
            if (elt.getAttributeValue("param_type").contains(
                    "[[integer untilv hole]]"))
            {
                cons_info = new ScSourceUntilvHole(uid, sketch);
            } else {
                cons_info = new ScSourceApplyHole(uid, sketch);
            }
        } else if (elt.getLocalName().equals("oracleapply")) {
            if (elt.getAttributeValue("param_type").contains(
                    "[[integer untilv oracle]]"))
            {
                cons_info = new ScSourceUntilvOracle(uid, sketch);
            } else {
                cons_info = new ScSourceApplyOracle(uid, sketch);
            }
            LineColumn start =
                    ScSourceLocation.fromXML(filename, entire_loc).start;
            eloc = new ScSourceLocation(filename, start.line);
        }
        if (cons_info == null) {
            DebugOut.assertFalse("no cons_info set.");
        }
        return new ScSourceConstruct(cons_info, eloc, ScSourceLocation.fromXML(
                filename, argument_loc));
    }

    public int compareTo(ScSourceConstruct other) {
        return entire_location.compareTo(other.entire_location);
    }
}
