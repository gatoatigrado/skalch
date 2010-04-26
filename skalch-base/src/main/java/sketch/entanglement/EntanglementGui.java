/*
 * @(#)HelloWorld.java 3.3 23-APR-04 Copyright (c) 2001-2004, Gaudenz Alder All rights
 * reserved. This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free Software
 * Foundation; either version 2.1 of the License, or (at your option) any later version.
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
 * should have received a copy of the GNU Lesser General Public License along with this
 * library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite
 * 330, Boston, MA 02111-1307 USA
 */
package sketch.entanglement;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JViewport;

import org.jgraph.JGraph;
import org.jgraph.graph.AttributeMap;
import org.jgraph.graph.DefaultEdge;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.DefaultGraphModel;
import org.jgraph.graph.GraphConstants;
import org.jgraph.graph.GraphModel;
import org.jgraph.graph.Port;

import com.jgraph.layout.JGraphFacade;
import com.jgraph.layout.organic.JGraphFastOrganicLayout;

public class EntanglementGui {

    private JGraph graph;
    private JFrame frame;
    private Map<DynAngel, DefaultGraphCell> angelToVertex;

    public EntanglementGui(EntanglementAnalysis ea) {
        angelToVertex = new HashMap<DynAngel, DefaultGraphCell>();

        initJGraph();

        // Show in Frame
        frame = new JFrame("Entanglement");
        frame.setSize(600, 600);
        frame.getContentPane().add(new JScrollPane(graph));
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        // frame.pack();
        frame.setVisible(true);

        addNodesVertices(ea, graph);
        layout(graph);
        // fitViewport(graph);

    }

    private void initJGraph() {
        // Switch off D3D because of Sun XOR painting bug
        // See http://www.jgraph.com/forum/viewtopic.php?t=4066
        System.setProperty("sun.java2d.d3d", "false");

        // Construct Model and Graph
        GraphModel model = new DefaultGraphModel();
        graph = new JGraph(model);

        // Control-drag should clone selection
        // graph.setCloneable(true);

        // Enable edit without final RETURN keystroke
        graph.setInvokesStopCellEditing(true);

        // When over a cell, jump to its default port (we only have one, anyway)
        graph.setJumpToDefaultPort(true);

        graph.setSize(600, 600);

    }

    public void addNodesVertices(EntanglementAnalysis ea, JGraph jgraph) {
        if (ea != null) {
            Set<DynAngel> allAngels = ea.getAllAngels();
            for (DynAngel angel : allAngels) {
                DefaultGraphCell cell =
                        new DefaultGraphCell(angel.toString(), new AttributeMap(
                                new HashMap()));
                GraphConstants.setBounds(cell.getAttributes(), new Rectangle2D.Double(
                        angel.staticAngelId * 200, angel.execNum * 200, 60, 20));
                cell.addPort();
                angelToVertex.put(angel, cell);
            }

            Set<DynAngelPair> entangledPairs = ea.getAllEntangledPairs();
            Set<DefaultEdge> edges = new HashSet<DefaultEdge>();
            for (DynAngelPair pair : entangledPairs) {
                // Create Edge
                DefaultEdge edge = new DefaultEdge(null, new AttributeMap(new HashMap()));
                DefaultGraphCell v1 = angelToVertex.get(pair.loc1);
                DefaultGraphCell v2 = angelToVertex.get(pair.loc2);

                // Fetch the ports from the new vertices, and connect them with the edge
                Port target, source;

                if (v1 instanceof Port) {
                    source = (Port) v1;
                } else {
                    source = (Port) jgraph.getModel().getChild(v1, 0);
                }

                if (v2 instanceof Port) {
                    target = (Port) v2;
                } else {
                    target = (Port) jgraph.getModel().getChild(v2, 0);
                }

                edge.setSource(source);
                edge.setTarget(target);
                edges.add(edge);
            }
            insertIntoGraph(jgraph, angelToVertex.values().toArray());

            Object[] cells = new Object[angelToVertex.size() + edges.size()];
            System.arraycopy(angelToVertex.values().toArray(), 0, cells, 0,
                    angelToVertex.size());
            System.arraycopy(edges.toArray(), 0, cells, angelToVertex.size(),
                    edges.size());

            insertIntoGraph(jgraph, cells);
        }
    }

    protected void insertIntoGraph(JGraph jgraph, Object[] cells) {
        // For performance, don't select inserted cells
        boolean selectsAll = jgraph.getGraphLayoutCache().isSelectsAllInsertedCells();
        boolean selectsLocal = jgraph.getGraphLayoutCache().isSelectsLocalInsertedCells();
        jgraph.getGraphLayoutCache().setSelectsAllInsertedCells(false);
        jgraph.getGraphLayoutCache().setSelectsLocalInsertedCells(false);

        jgraph.getModel().insert(cells, null, null, null, null);
        jgraph.getGraphLayoutCache().insert(cells);

        jgraph.getGraphLayoutCache().setSelectsAllInsertedCells(selectsAll);
        jgraph.getGraphLayoutCache().setSelectsLocalInsertedCells(selectsLocal);
    }

    protected void layout(JGraph jgraph) {
        JGraphFacade facade = new JGraphFacade(jgraph);
        JGraphFastOrganicLayout layout = new JGraphFastOrganicLayout();
        layout.setForceConstant(300);
        layout.run(facade);
        // Obtain a map of the resulting attribute changes from the facade
        Map nested = facade.createNestedMap(true, true);
        // Apply the results to the actual graph
        jgraph.getModel().edit(nested, null, null, null);

        jgraph.getGraphLayoutCache().edit(nested);

    }

    public static void fitViewport(JGraph graph) {
        int border = 5;
        Component parent = graph.getParent();
        if (parent instanceof JViewport) {
            Dimension size = ((JViewport) parent).getExtentSize();
            Rectangle2D p = graph.getCellBounds(graph.getRoots());
            if (p != null) {
                graph.setScale(Math.min(size.getWidth() /
                        (p.getX() + p.getWidth() + border), size.getHeight() /
                        (p.getY() + p.getHeight() + border)));
            }
        }
    }

    public static void main(String args[]) {
        List<Trace> traces = new ArrayList<Trace>();
        Trace t = new Trace();
        t.addEvent(0, 0, 2, 1);
        t.addEvent(0, 1, 3, 2);
        t.addEvent(0, 2, 3, 1);
        traces.add(t);
        t = new Trace();
        t.addEvent(0, 0, 2, 1);
        t.addEvent(0, 1, 3, 0);
        t.addEvent(0, 2, 3, 2);
        traces.add(t);
        t = new Trace();
        t.addEvent(0, 0, 2, 0);
        t.addEvent(0, 1, 3, 0);
        t.addEvent(0, 0, 3, 0);
        traces.add(t);
        EntanglementAnalysis ea = new EntanglementAnalysis(traces);

        EntanglementGui gui = new EntanglementGui(ea);

    }
}