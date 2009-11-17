package sketch.ui.queues;

public class TreeQueueIterator implements QueueIterator {

	private TreeQueueNode current;

	public TreeQueueIterator(TreeQueueNode root) {
		current = root;
	}

	public boolean checkValue(Object value) {
		if (current.children.containsKey(value)) {
			current = current.children.get(value);
			return true;
		} else {
			return false;
		}
	}

	public boolean canFinish() {
		return current.canFinish;
	}

}
