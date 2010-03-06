package choice.studies

import choice.test.LinkedList._
import choice.Choice._

/**
 * Satish's P3b
 * ------------
 * Sample trace: No solution found.
 */
object SatishList3b {

  def reverse(l: Node): Node = {
    var hist: List[Node] = Nil
    val len = listLength(l)
    var cur = l
    var prev = !![Node]
    assert(prev == l || (l.next != null && prev == l.next ) || prev == null)
    boundedWhile (len) (choiceBoolean()) {
      val x = !![Node]
      assert(x == cur || (cur != null && x == cur.next) || x == prev || (prev != null && x == prev.next))
      val y = !![Node]
      assert(y == cur || (cur != null && y == cur.next) || y == prev || (prev != null && y == prev.next))
      x.next = y
      hist = hist ::: List(x)
      prev = !![Node]
      cur = cur.next
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
