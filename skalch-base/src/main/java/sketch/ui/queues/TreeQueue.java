package sketch.ui.queues;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.Map.Entry;

import sketch.util.DebugOut;

public class TreeQueue implements Queue {

	TreeQueueNode root;
	private boolean finishedInserting;

	public TreeQueue() {
		root = new TreeQueueNode(null);
		finishedInserting = false;
	}

	public void insert(List<Object> trace) {
		if (!finishedInserting) {
			root.addChild(trace, 0);
		} else {
			DebugOut.print("Editing queue after accessing iterator.");
		}
	}

	public QueueIterator getIterator() {
		setFinishedInserts();
		return new TreeQueueIterator(root);
	}

	public void setFinishedInserts() {
		finishedInserting = true;
	}

	private List<? extends List<Object>> toListofLists() {
		Vector<Vector<Object>> listOfQueues = new Vector<Vector<Object>>();
		Vector<Object> queue = new Vector<Object>();
		createListOfQueues(root, queue, listOfQueues);
		return listOfQueues;
	}

	private void createListOfQueues(TreeQueueNode node, Vector<Object> queue,
			Vector<Vector<Object>> listOfQueues) {
		if (node.canFinish) {
			listOfQueues.add((Vector<Object>) queue.clone());
		}
		for (Iterator<Entry<Object, TreeQueueNode>> i = node.children
				.entrySet().iterator(); i.hasNext();) {
			Entry<Object, TreeQueueNode> child = i.next();
			queue.add(child.getKey());
			createListOfQueues(child.getValue(), queue, listOfQueues);
			queue.remove(queue.size() - 1);
		}
	}

	@Override
	public String toString() {
		return toListofLists().toString();
	}
}

class TreeQueueNode {

	final Object value;
	final Map<Object, TreeQueueNode> children;
	boolean canFinish;

	TreeQueueNode(Object v) {
		value = v;
		children = new HashMap<Object, TreeQueueNode>();
		canFinish = false;
	}

	void addChild(List<Object> path, int index) {
		TreeQueueNode child;
		if (index < path.size()) {
			Object nextValue = path.get(index);

			if (children.containsKey(nextValue)) {
				child = children.get(nextValue);
			} else {
				child = new TreeQueueNode(nextValue);
				children.put(nextValue, child);
			}

			child.addChild(path, index + 1);
		} else {
			canFinish = true;
		}
	}
}
