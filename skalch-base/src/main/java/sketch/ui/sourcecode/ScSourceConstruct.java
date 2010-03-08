package sketch.ui.sourcecode;

import nu.xom.Element;
import sketch.dyn.main.ScDynamicSketchCall;
import sketch.util.DebugOut;
import sketch.util.sourcecode.ScSourceLocation;
import sketch.util.sourcecode.ScSourceLocation.LineColumn;
import sketch.util.wrapper.XmlEltWrapper;
import sketch.util.wrapper.XmlNoXpathMatchException;

/**
 * A source construct info bound to a location in source.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class ScSourceConstruct implements Comparable<ScSourceConstruct> {
    public ScSourceConstructInfo constructInfo;
    /** location to print for identification only */
    public ScSourceLocation realLocation;
    public ScSourceLocation entireLocation;
    public ScSourceLocation argumentLocation;

    public ScSourceConstruct(ScSourceConstructInfo constructInfo,
            ScSourceLocation realLocation, ScSourceLocation entireLocation,
            ScSourceLocation argumentLocation)
    {
        this.realLocation = realLocation;
        this.constructInfo = constructInfo;
        this.entireLocation = entireLocation;
        this.argumentLocation = argumentLocation;
    }

    @Override
    public String toString() {
        return "ScSourceConstruct[entireLoc=" + entireLocation.toString() + ", argLoc=" +
                argumentLocation.toString() + "]";
    }

    public String getName() {
        return constructInfo.getName() + "@" + realLocation.start.toString();
    }

    public static ScSourceLocation getLocation(XmlEltWrapper root, String filename,
            String name, boolean canBeZeroLen)
    {
        try {
            XmlEltWrapper loc = root.XpathElt("rangepos[@name='" + name + "']");
            return ScSourceLocation.fromXML(filename, loc, canBeZeroLen);
        } catch (XmlNoXpathMatchException e) {
            XmlEltWrapper loc = root.XpathElt("position[@name='" + name + "']");
            LineColumn lc = LineColumn.fromXML(loc);
            return new ScSourceLocation(filename, lc.line);
        }
    }

    public static ScSourceConstruct fromNode(Element child_, String filename,
            ScDynamicSketchCall<?> sketchCall)
    {
        XmlEltWrapper elt = new XmlEltWrapper(child_);
        int uid = elt.intAttr("uid");
        ScSourceLocation eloc = getLocation(elt, filename, "entire_pos", false);
        ScSourceLocation rloc = eloc;
        String pt = elt.getAttributeValue("param_type");
        // vars for individual cases to set
        ScSourceConstructInfo consInfo = null;
        boolean zeroLenArgLoc = false;
        if (elt.getLocalName().equals("holeapply")) {
            if (pt.contains("[[integer untilv hole]]")) {
                consInfo = new ScSourceUntilvHole(uid, sketchCall);
            } else {
                DebugOut.assertSlow(pt.contains("[[object apply hole]]"),
                        "unknown parameter type", pt);
                consInfo = new ScSourceApplyHole(uid, sketchCall);
            }
        } else if (elt.getLocalName().equals("oracleapply")) {
            if (pt.contains("[[integer untilv oracle]]")) {
                consInfo = new ScSourceUntilvOracle(uid, sketchCall);
            } else if (pt.contains("[[boolean oracle]]")) {
                zeroLenArgLoc = true;
                consInfo = new ScSourceBooleanOracle(uid, sketchCall);
            } else {
                DebugOut.assertSlow(pt.contains("[[object apply oracle]]"),
                        "unknown parameter type", pt);
                consInfo = new ScSourceApplyOracle(uid, sketchCall);
            }
            eloc = new ScSourceLocation(filename, eloc.start.line);
        }
        if (consInfo == null) {
            DebugOut.assertFalse("no consInfo set.");
        }
        ScSourceLocation argLoc = getLocation(elt, filename, "arg_pos", zeroLenArgLoc);
        return new ScSourceConstruct(consInfo, rloc, eloc, argLoc);
    }

    public int compareTo(ScSourceConstruct other) {
        return entireLocation.compareTo(other.entireLocation);
    }
}
