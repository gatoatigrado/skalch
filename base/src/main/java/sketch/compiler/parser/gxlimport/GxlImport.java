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
import sketch.compiler.ast.core.StreamSpec;
import sketch.compiler.ast.core.typs.TypeStruct;
import sketch.util.DefaultHashMap;

/**
 * Import all nodes from a GXL file and create a Program instance.
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class GxlImport {
    public DefaultHashMap<String, Vector<GXLNode>> nodes_by_type;
    public HashMap<String, GXLNode> nodes_by_id;
    public DefaultHashMap<String, Vector<GXLEdge>> edges_by_source_id;
    public DefaultHashMap<String, Vector<GXLEdge>> edges_by_target_id;
    public Vector<TypeStruct> structs;
    public Vector<StreamSpec> streams;

    protected void init(GXLGraph graph) {
        structs = new Vector<TypeStruct>();
        streams = new Vector<StreamSpec>();
        nodes_by_type = new DefaultHashMap<String, Vector<GXLNode>>();
        nodes_by_type.defvalue = nodes_by_type.new DefValueGenerator() {
            public Vector<GXLNode> get_value() {
                return new Vector<GXLNode>();
            }
        };
        nodes_by_id = new HashMap<String, GXLNode>();
        edges_by_source_id = new DefaultHashMap<String, Vector<GXLEdge>>();
        edges_by_target_id = new DefaultHashMap<String, Vector<GXLEdge>>();
        edges_by_source_id.defvalue =
                edges_by_target_id.defvalue =
                        edges_by_target_id.new DefValueGenerator() {
                            public Vector<GXLEdge> get_value() {
                                return new Vector<GXLEdge>();
                            }
                        };
        for (int a = 0; a < graph.getGraphElementCount(); a++) {
            GXLGraphElement elt = graph.getGraphElementAt(a);
            if (elt instanceof GXLNode) {
                GXLNode elt2 = (GXLNode) elt;
                nodes_by_type.get(nodeType(elt2)).add(elt2);
                nodes_by_id.put(elt2.getID(), elt2);
            } else if (elt instanceof GXLEdge) {
                GXLEdge elt2 = (GXLEdge) elt;
                edges_by_source_id.get(elt2.getSourceID()).add(elt2);
                edges_by_target_id.get(elt2.getTargetID()).add(elt2);
            }
        }
        for (GXLNode pkg : nodes_by_type.get("PackageDef")) {
            handleNode(pkg);
        }
    }

    private void handleNode(GXLNode pkg) {
    }

    public GxlImport(GXLGraph graph) {
        init(graph);
    }

    public GxlImport(File graph_file) {
        GXLDocument doc = null;
        try {
            doc = new GXLDocument(graph_file);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            System.exit(1);
        }
        GXLGXL gxl = doc.getDocumentElement();
        for (int i = 0; i < gxl.getGraphCount(); i++) {
            GXLGraph graph = gxl.getGraphAt(i);
            if (graph.getID() == "DefaultGraph") {
                init(graph);
                return;
            }
        }
        assert false : "couldn't find a graph named 'DefaultGraph'";
    }

    public static String nodeType(GXLNode node) {
        String uri = node.getType().getURI().toString();
        assert uri.startsWith("#") : "node type not relative " + uri;
        return uri.substring(1);
    }
}
