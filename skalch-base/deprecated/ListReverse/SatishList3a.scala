package choice.studies

import choice.test.LinkedList._
import choice.Choice._

/**
 * Satish's P3a
 * ------------
 * Sample trace: No solution found.
 * Note that this finds no solution fairly quickly (as opposed to previous versions) since of course the asserts cut down on the search space.
 * Also note that the !! values here and in the following versions are really ?? uses
 */
object SatishList3a {

  def reverse(l: Node): Node = {
    var hist: List[Node] = Nil
    val len = listLength(l)
    var cur = l
    boundedWhile (len + 2) (choiceBoolean()) {
      val x = !![Node]
      // Note that I added a null check here
      assert(x == cur || (cur != null && x == cur.next))
      val y = !![Node]
      assert(y == cur || (cur != null && y == cur.next))
      x.next = y
      hist = hist ::: List(x)
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
