package sketch.queues;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.List;
import java.util.Vector;

import sketch.util.DebugOut;

public class QueueFileInput {

	Queue queue;

	public QueueFileInput(String fileName) {
		queue = createTreeQueueFromList(readQueuesFromFile(fileName));
	}

	public Queue getQueue() {
		queue.setFinishedInserts();
		return queue;
	}

	protected List<? extends List<Object>> readQueuesFromFile(String fileName) {
		Vector<Vector<Object>> listOfQueues = new Vector<Vector<Object>>();
		try {
			ObjectInputStream input = new ObjectInputStream(
					new FileInputStream(fileName));
			listOfQueues = (Vector<Vector<Object>>) input.readObject();
		} catch (FileNotFoundException e) {
			DebugOut.print_exception("Problem reading file " + fileName
					+ " for queues", e);
		} catch (IOException e) {
			DebugOut.print_exception("Problem opening file " + fileName
					+ " for queues", e);
		} catch (ClassNotFoundException e) {
			DebugOut.print_exception("Queue file input in wrong input", e);
		}
		return listOfQueues;
	}

	protected Queue createTreeQueueFromList(List<? extends List<Object>> queues) {
		Queue queue = new TreeQueue();
		for (List<Object> trace : queues) {
			queue.insert(trace);
		}
		return queue;
	}

	public static void main(String args[]) {
		@SuppressWarnings("unused")
		QueueFileInput input = new QueueFileInput(
				"/home/shaon/Research/workspace/skalch/skalch-base/redblack.out");
	}

}
