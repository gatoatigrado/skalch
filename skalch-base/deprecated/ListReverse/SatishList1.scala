package choice.studies

import choice.test.LinkedList._
import choice.Choice._

/**
 * Satish's P1
 * -----------
 * Sample trace: true, Node 3, Node 3, true, Node 3, Node 3, true, Node 3, Node 2, true, Node 2, Node 1, true, Node 1, null, Node 3
 */
object SatishList1 {

  def reverse(l: Node): Node = {
    val len = listLength(l)
    // I use boundedWhile so the loop terminates.  The maximum number is somewhat arbitrary: at most we could do one thing per each element and maybe some extra work at the beginning and the end.
    boundedWhile (len + 2) (choiceBoolean()) {
      !![Node].next = !![Node]
    }
    return !![Node]
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
