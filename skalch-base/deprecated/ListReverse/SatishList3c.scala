package choice.studies

import choice.test.LinkedList._
import choice.Choice._

/**
 * Satish's P3c
 * ------------
 * Sample trace: null, true, Node 1, null, Node 1, true, Node 2, Node 1, Node 2, true, Node 3, Node 2, null, false, Node 3.
 * This solution took me about one minute to find (longer than earlier ones).
 */
object SatishList3c {

  def reverse(l: Node): Node = {
    var hist: List[Node] = Nil
    val len = listLength(l)
    var cur = l
    var prev = !![Node]
    assert(prev == l || (l.next != null && prev == l.next ) || prev == null)
    boundedWhile (len + 2) (choiceBoolean()) {
      val x = !![Node]
      assert(x == cur || (cur != null && x == cur.next) || x == prev || (prev != null && x == prev.next))
      val y = !![Node]
      assert(y == cur || (cur != null && y == cur.next) || y == prev || (prev != null && y == prev.next))
      // Moved this here before we overwrite it.
      // Note that we don't need a null check here since if we hit that case we backtrack and use a shorter loop.
      cur = cur.next
      x.next = y
      hist = hist ::: List(x)
      prev = !![Node]
    }
    assertIsCorrectOrder(hist)
    return !![Node]
  }

  // Note that I use <= instead of < here.
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
