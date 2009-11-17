package sketch.ui.queues;

import java.util.List;

public interface Queue {

	void insert(List<Object> trace);

	QueueIterator getIterator();

	void setFinishedInserts();
}
