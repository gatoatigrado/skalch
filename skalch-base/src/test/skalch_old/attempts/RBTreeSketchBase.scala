package test.skalch_old.attempts

import scala.collection.mutable.HashSet

import skalch.DynamicSketch

abstract class RBTreeSketchBase extends DynamicSketch {
    /** nodes only link to their children. the algorithm should
     * maintain a stack of parents when traversing the tree.
     */
    class RBTreeNode(var isBlack : Boolean, var value : Int) {
        var leftChild : RBTreeNode = null
        var rightChild : RBTreeNode = null
        /** assumes ident was already inserted. responsible for the newline */
        def formatTree(ident_ : String) : String = {
            val ident = ident_ + "    "
            val lcString = if (leftChild == null) "null\n" else leftChild.formatTree(ident)
            val rcString = if (rightChild == null) "null\n" else rightChild.formatTree(ident)
            toString() + "\n" + ident + "< " + lcString + ident + "> " + rcString
        }
        override def toString() = (if (isBlack) "B" else "R") + "(" + value + ")"

        // sketch-related
        def checkNumBlack() : Int = {
            val num_left = if (leftChild == null) 0 else leftChild.checkNumBlack()
            val num_right = if (rightChild == null) 0 else rightChild.checkNumBlack()
            synthAssertTerminal(num_left == num_right)
            num_left + num_right + (if (isBlack) 1 else 0)
        }
        /** returns isBlack for convenience */
        def checkRedChildrenBlack() : Boolean = {
            val leftBlack = if (leftChild == null) true
                else leftChild.checkRedChildrenBlack()
            val rightBlack = if (rightChild == null) true
                else rightChild.checkRedChildrenBlack()
            synthAssertTerminal(isBlack || (leftBlack && rightBlack))
            isBlack
        }
        /** number of nodes reachable including this one */
        def numNodes() : Int = {
            (if (leftChild == null) 0 else leftChild.numNodes()) +
            (if (rightChild == null) 0 else rightChild.numNodes()) +
            1
        }

        /** check that the tree is actually a tree, and node order is correct. */
        private def checkIsTreeInner(leftBound : Int, rightBound : Int) : Int = {
            RBTreeVisitedList.put(this)
            if (leftChild != null) {
                val left_child = leftChild.checkIsTreeInner(leftBound, value)
                synthAssertTerminal(leftBound < left_child && left_child <= value)
            }
            if (rightChild != null) {
                val right_child = rightChild.checkIsTreeInner(value, rightBound)
                synthAssertTerminal(value <= right_child && right_child < rightBound)
            }
            value
        }
        def checkIsTree() {
            RBTreeVisitedList.reset()
            checkIsTreeInner(-1 << 30, 1 << 30)
        }
    }

    object RBTreeVisitedList {
        val visitedNodes = new HashSet[RBTreeNode]()
        def reset() { visitedNodes.clear() }
        def put(node : RBTreeNode) {
            synthAssertTerminal(visitedNodes.add(node))
        }
    }
}
