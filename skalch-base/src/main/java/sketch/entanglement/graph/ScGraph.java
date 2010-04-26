package sketch.entanglement.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ScGraph<T> {

    private Map<T, Set<T>> vertices;

    public ScGraph() {
        vertices = new HashMap<T, Set<T>>();
    }

    public void addVertex(T vertex) {
        vertices.put(vertex, new HashSet<T>());
    }

    public void addEdge(T v1, T v2) {
        if (!vertices.containsKey(v1)) {
            addVertex(v1);
        }

        if (!vertices.containsKey(v2)) {
            addVertex(v2);
        }

        vertices.get(v1).add(v2);
        vertices.get(v2).add(v1);
    }

    public Set<Set<T>> getConnectedComponents() {
        Set<Set<T>> connectedComponents = new HashSet<Set<T>>();
        List<T> vertexList = new ArrayList<T>(vertices.keySet());
        while (!vertexList.isEmpty()) {
            Set<T> newCC = new HashSet<T>();
            Set<T> addedVertices = new HashSet<T>();

            connectedComponents.add(newCC);
            T firstVertex = vertexList.remove(0);
            newCC.add(firstVertex);
            addedVertices.add(firstVertex);

            while (addedVertices.size() > 0) {
                addedVertices.clear();
                for (T element : newCC) {
                    Set<T> connectedVertices = vertices.get(element);
                    for (T nextVertex : connectedVertices) {
                        if (!newCC.contains(nextVertex)) {
                            addedVertices.add(nextVertex);
                            vertexList.remove(nextVertex);
                        }
                    }
                }
                newCC.addAll(addedVertices);
            }
        }
        return connectedComponents;
    }

    @Override
    public String toString() {
        String returnString = "";
        for (T node : vertices.keySet()) {
            returnString += ("(" + node + ":");
            for (T edge : vertices.get(node)) {
                returnString += edge + ",";
            }
            returnString += ")";
        }
        return returnString;
    }
}