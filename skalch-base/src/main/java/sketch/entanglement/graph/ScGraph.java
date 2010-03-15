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

    public List<List<T>> getConnectedComponents() {
        List<List<T>> connectedComponents = new ArrayList<List<T>>();
        List<T> vertexList = new ArrayList<T>(vertices.keySet());
        while (!vertexList.isEmpty()) {
            List<T> newCC = new ArrayList<T>();
            List<T> addedVertices = new ArrayList<T>();

            connectedComponents.add(newCC);
            T firstVertex = vertexList.remove(0);
            newCC.add(firstVertex);
            addedVertices.add(firstVertex);

            while (addedVertices.size() > 0) {
                addedVertices.clear();
                for (T element : newCC) {
                    Set<T> connectedVertices = vertices.get(element);
                    // System.out.println(connectedVertices.size());
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
}