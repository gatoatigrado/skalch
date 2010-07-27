package sketch.compiler.parser.gxlimport;

import java.io.File;
import java.util.HashMap;
import java.util.Vector;

import net.sourceforge.gxl.GXLAtomicValue;
import net.sourceforge.gxl.GXLDocument;
import net.sourceforge.gxl.GXLEdge;
import net.sourceforge.gxl.GXLGXL;
import net.sourceforge.gxl.GXLGraph;
import net.sourceforge.gxl.GXLGraphElement;
import net.sourceforge.gxl.GXLNode;
import net.sourceforge.gxl.GXLValue;
import sketch.compiler.ast.core.Program;
import sketch.compiler.ast.core.StreamSpec;
import sketch.compiler.ast.core.typs.TypeStruct;
import sketch.util.DebugOut;
import sketch.util.DefaultVectorHashMap;

/**
 * Import all nodes from a GXL file and create a Program instance.
 *
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class GxlImport {
    public DefaultVectorHashMap<String, GXLNode> nodes_by_type;
    public HashMap<String, GXLNode> nodes_by_id;
    public DefaultVectorHashMap<NodeStringTuple, GXLEdge> edges_by_source;
    public DefaultVectorHashMap<NodeStringTuple, GXLEdge> edges_by_target;
    public Vector<TypeStruct> structs;
    public Vector<StreamSpec> streams;
    public GxlHandleNodes handler;
    protected final GxlSketchOptions args;

    /** import, the start the SequentialSketchGxlMain */
    protected void init(final GXLGraph graph) {
        DebugOut.print("    initializing...");
        this.handler = new GxlHandleNodes(this);
        this.structs = new Vector<TypeStruct>();
        this.streams = new Vector<StreamSpec>();
        this.nodes_by_type = new DefaultVectorHashMap<String, GXLNode>();
        this.nodes_by_id = new HashMap<String, GXLNode>();
        this.edges_by_source = new DefaultVectorHashMap<NodeStringTuple, GXLEdge>();
        this.edges_by_target = new DefaultVectorHashMap<NodeStringTuple, GXLEdge>();
        for (int a = 0; a < graph.getGraphElementCount(); a++) {
            GXLGraphElement elt = graph.getGraphElementAt(a);
            if (elt instanceof GXLNode) {
                GXLNode elt2 = (GXLNode) elt;
                this.nodes_by_type.get(GxlImport.nodeType(elt2)).add(elt2);
                this.nodes_by_id.put(elt2.getID(), elt2);
            } else if (elt instanceof GXLEdge) {
                GXLEdge elt2 = (GXLEdge) elt;
                final String etyp = GxlImport.edgeType(elt2);
                final GXLNode source = (GXLNode) elt2.getSource();
                final GXLNode target = (GXLNode) elt2.getTarget();
                NodeStringTuple srckey = new NodeStringTuple(source, etyp);
                NodeStringTuple tgtkey = new NodeStringTuple(target, etyp);
                this.edges_by_source.get(srckey).add(elt2);
                this.edges_by_target.get(tgtkey).add(elt2);
            }
        }
        DebugOut.print("done categorizing nodes...");
        if (this.nodes_by_type.get("PackageDef").isEmpty()) {
            DebugOut.assertFalse("couldn't find any packagedefs.");
        }
        for (GXLNode pkg : this.nodes_by_type.get("PackageDef")) {
            Program prog = this.handler.getProgram(pkg);
            // (new SimpleCodePrinter()).visitProgram(prog);
            if (!this.args.gxlOpts.noDefaults) {
                this.args.feOpts.keepTmp = true;
                this.args.feOpts.keepAsserts = true;
            }
            if (this.args.gxlOpts.dumpInputParse != null) {
                prog.debugDump(new File(this.args.gxlOpts.dumpInputParse));
            }
            prog.debugDump("input parse");
            (new SequentialSketchGxlMain(this.args, prog)).run();
        }
    }

    public GxlImport(final GXLGraph graph, final GxlSketchOptions args) {
        this.args = args;
        this.init(graph);
    }

    public GxlImport(final GxlSketchOptions args) {
        this.args = args;
        GXLDocument doc = null;
        try {
            doc = new GXLDocument(args.sketchFile);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        GXLGXL gxl = doc.getDocumentElement();
        for (int i = 0; i < gxl.getGraphCount(); i++) {
            GXLGraph graph = gxl.getGraphAt(i);
            if (graph.getID().equals("DefaultGraph")) {
                this.init(graph);
                return;
            }
        }
        DebugOut.assertFalse("couldn't find a graph named 'DefaultGraph'");
    }

    public static String nodeType(final GXLNode node) {
        String uri = node.getType().getURI().toString();
        assert uri.startsWith("#") : "node type not relative " + uri;
        String result = uri.substring(1);
        if (!result.equals(result.trim())) {
            DebugOut.assertFalse("type name has trailing whitespace characters.");
        }
        return result;
    }

    public static String edgeType(final GXLEdge edge) {
        String uri = edge.getType().getURI().toString();
        assert uri.startsWith("#") : "node type not relative " + uri;
        String result = uri.substring(1);
        if (!result.equals(result.trim())) {
            DebugOut.assertFalse("type name has trailing whitespace characters.");
        }
        return result;
    }

    public static void main(final String[] args) {
        final GxlSketchOptions options = new GxlSketchOptions(args);
        if (options.args.length != 1) {
            DebugOut.assertFalse("wrong usage: only 1 argument expected");
        }
        new GxlImport(options);
    }

    public void debugPrintNode(final String msg, final GXLNode node) {
        DebugOut.fmt("\n=== %s - printing info about node '%s' ===", msg, node.getID());
        for (int a = 0; a < node.getAttrCount(); a++) {
            DebugOut.fmt("    attr['%s'] = %s", node.getAttrAt(a).getName(),
                    this.attrAsString(node.getAttrAt(a).getValue()));
        }
    }

    public String attrAsString(final GXLValue value) {
        if (value instanceof GXLAtomicValue) {
            return ((GXLAtomicValue) value).getValue();
        } else {
            DebugOut.print("don't know what to do with value", value);
            throw new RuntimeException();
        }
    }

    public void debugPrintEdge(final String msg, final GXLEdge edge) {
        DebugOut.fmt("\n=== %s - printing info about edge '%s' ===", msg, edge.getID());
        DebugOut.fmt("source: '%s', type", edge.getSourceID(),
                GxlImport.nodeType((GXLNode) edge.getSource()));
        DebugOut.fmt("target: '%s', type", edge.getTargetID(),
                GxlImport.nodeType((GXLNode) edge.getTarget()));
        for (int a = 0; a < edge.getAttrCount(); a++) {
            DebugOut.fmt("    attr['%s'] = %s", edge.getAttrAt(a).getName(),
                    this.attrAsString(edge.getAttrAt(a).getValue()));
        }
    }
}
