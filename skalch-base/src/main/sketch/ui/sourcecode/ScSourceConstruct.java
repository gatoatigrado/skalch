package sketch.ui.sourcecode;

import nu.xom.Element;
import sketch.dyn.ScDynamicSketch;
import sketch.util.DebugOut;
import sketch.util.XmlEltWrapper;
import sketch.util.XmlNoXpathMatchException;
import sketch.util.sourcecode.ScSourceLocation;
import sketch.util.sourcecode.ScSourceLocation.LineColumn;

/**
 * A source construct info bound to a location in source.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceConstruct implements Comparable<ScSourceConstruct> {
    public ScSourceConstructInfo construct_info;
    /** location to print for identification only */
    public ScSourceLocation real_location;
    public ScSourceLocation entire_location;
    public ScSourceLocation argument_location;

    public ScSourceConstruct(ScSourceConstructInfo construct_info,
            ScSourceLocation real_location, ScSourceLocation entire_location,
            ScSourceLocation argument_location)
    {
        this.real_location = real_location;
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
        return construct_info.getName() + "@" + real_location.start.toString();
    }

    public static ScSourceLocation get_location(XmlEltWrapper root,
            String filename, String name, boolean can_be_zero_len)
    {
        try {
            XmlEltWrapper loc = root.XpathElt("rangepos[@name='" + name + "']");
            return ScSourceLocation.fromXML(filename, loc, can_be_zero_len);
        } catch (XmlNoXpathMatchException e) {
            XmlEltWrapper loc = root.XpathElt("position[@name='" + name + "']");
            LineColumn lc = LineColumn.fromXML(loc);
            return new ScSourceLocation(filename, lc.line);
        }
    }

    public static ScSourceConstruct from_node(Element child_, String filename,
            ScDynamicSketch sketch)
    {
        XmlEltWrapper elt = new XmlEltWrapper(child_);
        int uid = elt.int_attr("uid");
        ScSourceLocation eloc =
                get_location(elt, filename, "entire_pos", false);
        ScSourceLocation rloc = eloc;
        String pt = elt.getAttributeValue("param_type");
        // vars for individual cases to set
        ScSourceConstructInfo cons_info = null;
        boolean zero_len_arg_loc = false;
        if (elt.getLocalName().equals("holeapply")) {
            if (pt.contains("[[integer untilv hole]]")) {
                cons_info = new ScSourceUntilvHole(uid, sketch);
            } else {
                DebugOut.assertSlow(pt.contains("[[object apply hole]]"),
                        "unknown parameter type", pt);
                cons_info = new ScSourceApplyHole(uid, sketch);
            }
        } else if (elt.getLocalName().equals("oracleapply")) {
            if (pt.contains("[[integer untilv oracle]]")) {
                cons_info = new ScSourceUntilvOracle(uid, sketch);
            } else if (pt.contains("[[boolean oracle]]")) {
                zero_len_arg_loc = true;
                cons_info = new ScSourceBooleanOracle(uid, sketch);
            } else {
                DebugOut.assertSlow(pt.contains("[[object apply oracle]]"),
                        "unknown parameter type", pt);
                cons_info = new ScSourceApplyOracle(uid, sketch);
            }
            eloc = new ScSourceLocation(filename, eloc.start.line);
        }
        if (cons_info == null) {
            DebugOut.assertFalse("no cons_info set.");
        }
        ScSourceLocation arg_loc =
                get_location(elt, filename, "arg_pos", zero_len_arg_loc);
        return new ScSourceConstruct(cons_info, rloc, eloc, arg_loc);
    }

    public int compareTo(ScSourceConstruct other) {
        return entire_location.compareTo(other.entire_location);
    }
}
