package sketch.ui.sourcecode;

import nu.xom.Element;
import sketch.dyn.ScDynamicSketch;
import sketch.util.DebugOut;
import sketch.util.XmlEltWrapper;

/**
 * A source construct info bound to a location in source.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScSourceConstruct {
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

    public static ScSourceConstruct from_node(Element child_, String filename,
            ScDynamicSketch sketch)
    {
        XmlEltWrapper elt = new XmlEltWrapper(child_);
        if (!elt.getLocalName().equals("sketchconstruct")) {
            DebugOut
                    .assertFalse("call from_node on sketchconstruct tags only.");
        }
        if (elt.getAttributeValue("type").equals("$qmark$qmark")) {
            DebugOut.print("bang bang");
            int uid = elt.int_attr("uid");
            // "rangepos[@name='entire_position']"
            // "rangepos[@name='argument_position']"
            XmlEltWrapper entire_loc =
                    elt.XpathElt("rangepos[@name='entire_position']");
            XmlEltWrapper argument_loc =
                    elt.XpathElt("rangepos[@name='argument_position']");
            return new ScSourceConstruct(new ScSourceDynamicHole(uid, sketch),
                    ScSourceLocation.fromXML(filename, entire_loc),
                    ScSourceLocation.fromXML(filename, argument_loc));
            // return new ScSourceConstruct(new ScSourceDynamicHole(uid,
            // sketch));
        }
        DebugOut.assertFalse("unsupported construct info.");
        return null;
    }
}
