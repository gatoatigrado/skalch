package sketch.compiler.parser.gxlimport;

import java.io.File;
import java.util.HashMap;
import java.util.Vector;

import net.sourceforge.gxl.GXLDocument;
import net.sourceforge.gxl.GXLEdge;
import net.sourceforge.gxl.GXLGXL;
import net.sourceforge.gxl.GXLGraph;
import net.sourceforge.gxl.GXLGraphElement;
import net.sourceforge.gxl.GXLNode;
import scala.Tuple2;
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
    public DefaultVectorHashMap<Tuple2<GXLNode, String>, GXLEdge> edges_by_source;
    public DefaultVectorHashMap<Tuple2<GXLNode, String>, GXLEdge> edges_by_target;
    public Vector<TypeStruct> structs;
    public Vector<StreamSpec> streams;
    public GxlHandleNodes handler;

    protected void init(final GXLGraph graph) {
        DebugOut.print("    initializing...");
        this.handler = new GxlHandleNodes(this);
        this.structs = new Vector<TypeStruct>();
        this.streams = new Vector<StreamSpec>();
        this.nodes_by_type = new DefaultVectorHashMap<String, GXLNode>();
        this.nodes_by_id = new HashMap<String, GXLNode>();
        this.edges_by_source =
                new DefaultVectorHashMap<Tuple2<GXLNode, String>, GXLEdge>();
        this.edges_by_target =
                new DefaultVectorHashMap<Tuple2<GXLNode, String>, GXLEdge>();
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
                final Tuple2<GXLNode, String> srckey =
                        new Tuple2<GXLNode, String>(source, etyp);
                final Tuple2<GXLNode, String> tgtkey =
                        new Tuple2<GXLNode, String>(target, etyp);
                this.edges_by_source.get(srckey).add(elt2);
                this.edges_by_source.get(tgtkey).add(elt2);
            }
        }
        DebugOut.print("done categorizing nodes...");
        if (this.nodes_by_type.get("PackageDef").isEmpty()) {
            DebugOut.assertFalse("couldn't find any packagedefs.");
        }
        for (GXLNode pkg : this.nodes_by_type.get("PackageDef")) {
            Program prog = this.handler.getProgram(pkg);
            System.out.println("got program " + prog);
        }
    }

    public GxlImport(final GXLGraph graph) {
        this.init(graph);
    }

    public GxlImport(final File graph_file) {
        GXLDocument doc = null;
        try {
            doc = new GXLDocument(graph_file);
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

    /** tmp main fcn */
    public static void main(final String[] args) {
        if (args.length != 1) {
            System.err.println("usage: gxlimport gxlfile");
        } else {
            System.out.println("creating program from file " + args[0]);
            new GxlImport(new File(args[0]));
        }
    }
}
