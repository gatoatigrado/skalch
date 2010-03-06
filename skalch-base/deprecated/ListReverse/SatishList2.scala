package choice.studies

import choice.test.LinkedList._
import choice.Choice._

/**
 * Satish's P2
 * -----------
 * Sample trace: true, Node 1, null, true, Node 1, null, true, Node 1, null, true, Node 2, Node 1, true, Node 3, Node 2, Node 3
 */
object SatishList2 {

  def reverse(l: Node): Node = {
    var hist: List[Node] = Nil
    val len = listLength(l)
    boundedWhile (len + 2) (choiceBoolean()) {
      val x = !![Node]
      x.next = !![Node]
      hist = hist ::: List(x)
    }
    assertIsCorrectOrder(hist)
    return !![Node]
  }

  // Note that I use <= instead of < here.  Perhaps we'll need multiple steps at the same node as a corner case?
  def assertIsCorrectOrder(hist: List[Node]): Unit = {
    hist.foldLeft (0) { (prev, cur) => assert(prev <= cur.value); cur.value }
  }

  def main(args: Array[String]) {
    if (args.length != 1) {
      System.err.println("Please give me one argument, the length of the list.")
      System.exit(2)
    }
    val n = Integer.parseInt(args(0))
    run(testReverse(n, buildList(n), reverse))
  }

}
