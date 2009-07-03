package test

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

        /** check that the tree is actually a tree -- i.e. contains no circular links */
        private def checkIsTreeInner() {
            RBTreeVisitedList.put(this)
            if (leftChild != null) { leftChild.checkIsTree() }
            if (rightChild != null) { rightChild.checkIsTree() }
        }
        def checkIsTree() {
            RBTreeVisitedList.reset()
            checkIsTreeInner()
        }
    }

    object RBTreeVisitedList {
        var length : Int = 0
        val visitedNodes = new Array[RBTreeNode](100)
        def reset() { length = 0 }
        def put(node : RBTreeNode) {
            for (i <- 0 until length) {
                synthAssertTerminal(!visitedNodes(i).eq(node))
            }
            visitedNodes(length) = node
            length += 1
        }
    }
}
