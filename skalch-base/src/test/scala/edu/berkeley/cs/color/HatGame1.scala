
package edu.berkeley.cs.color

import scala.collection.immutable.HashMap
import skalch.AngelicSketch
import sketch.util._

class HatGame1Sketch extends AngelicSketch {
    val tests = Array(())

    def main() {
        val num: Int = 3

        // stores the mapping from (hat mapping, player id) to color 
        var colorMap = Map[(List[Int], String), java.awt.Color]();

        // iterates through all combinations of numVal hats ranging from [0,maxVal-1]
        class HatIterator(val numVal: Int, val maxVal: Int) {
            var nextHat: Array[Int] = new Array[Int](numVal)
            for (i <- 0 until nextHat.length) {
                nextHat(i) = 0
            }

            def peek(): List[Int] = {
                var returnVal: List[Int] = Nil
                if (nextHat != null) {
                    for (i <- 0 until nextHat.length) {
                        returnVal ::= nextHat(i)
                    }
                }
                return returnVal
            }

            def next(): List[Int] = {
                var returnVal = peek()
                if (returnVal != Nil) {
                    nextHat(0) += 1
                    for (i <- 0 until nextHat.length) {
                        if (nextHat(i) >= maxVal) {
                            if (i == nextHat.length - 1) {
                                nextHat = null
                            } else {
                                nextHat(i) = 0
                                nextHat(i + 1) += 1
                            }
                        } else {
                            return returnVal
                        }
                    }
                }
                return returnVal
            }
        }

        // returns an angelic player function. the table for the player's choices
        // is filled in angellically in real time
        def getFunction(name: String): List[Int] => Int = {
            var output: List[Int] = Nil
            for (value <- 0 until num) {
                output ::= value
            }

            var input: List[List[Int]] = Nil
            val iterator: HatIterator = new HatIterator(num - 1, num)
            var value: List[Int] = iterator.next
            while (value != Nil) {
                input ::= value
                value = iterator.next
            }

            var valueMap: Map[List[Int], Int] = new HashMap[List[Int], Int]

            val function: List[Int] => Int = (input => {
                if (!valueMap.contains(input)) {
                    valueMap += input -> !!(output)
                    val c = sklast_angel_color()
                    colorMap += (input, name) -> c
                    skdprint("Creating input: " + input + " -> " + valueMap(input), c)
                }
                valueMap(input)
            })
            return function
        }

        // returns a list of player functions
        def getGuessingFunction(): List[(List[Int] => Int)] = {
            var guessingFunctions: List[(List[Int] => Int)] = Nil
            for (i <- 0 until num) {
                guessingFunctions ::= getFunction(i.toString)
            }
            return guessingFunctions
        }

        // takes a table and returns a player function
        def getConstantFunction(valueMap: Map[List[Int], Int]): List[Int] => Int = {
            return (input => valueMap(input))
        }

        def getConstantGuessingFunction(): List[(List[Int] => Int)] = {
            var guessingFunctions: List[(List[Int] => Int)] = Nil

            var valueMap: Map[List[Int], Int] = new HashMap[List[Int], Int]()
                  valueMap += List(0,0) -> 2
                  valueMap += List(0,1) -> 1
                  valueMap += List(0,2) -> 0
                  valueMap += List(1,0) -> 0
                  valueMap += List(1,1) -> 2
                  valueMap += List(1,2) -> 1
                  valueMap += List(2,0) -> 1
                  valueMap += List(2,1) -> 0
                  valueMap += List(2,2) -> 2
            
//                  valueMap += List(0,0) -> 2
//                  valueMap += List(0,1) -> 2
//                  valueMap += List(0,2) -> 1
//                  valueMap += List(1,0) -> 0
//                  valueMap += List(1,1) -> 2
//                  valueMap += List(1,2) -> 0
//                  valueMap += List(2,0) -> 1
//                  valueMap += List(2,1) -> 0
//                  valueMap += List(2,2) -> 1
                  
//                  valueMap += List(0,0) -> 0
//                  valueMap += List(0,1) -> 1
//                  valueMap += List(0,2) -> 2
//                  valueMap += List(1,0) -> 1
//                  valueMap += List(1,1) -> 2
//                  valueMap += List(1,2) -> 0
//                  valueMap += List(2,0) -> 2
//                  valueMap += List(2,1) -> 0
//                  valueMap += List(2,2) -> 1

//            valueMap += List(0, 0) -> 0
//            valueMap += List(0, 1) -> 2
//            valueMap += List(0, 2) -> 1
//            valueMap += List(1, 0) -> 2
//            valueMap += List(1, 1) -> 1
//            valueMap += List(1, 2) -> 0
//            valueMap += List(2, 0) -> 1
//            valueMap += List(2, 1) -> 0
//            valueMap += List(2, 2) -> 2

            for (i <- 0 until num - 1) {
                guessingFunctions ::= getFunction("" + i)
            }
            guessingFunctions ::= getConstantFunction(valueMap)
            return guessingFunctions
        }

        // ensures the player functions will work on any hat combination
        def checkGuesses(guessingFunction: List[(List[Int] => Int)]) {
            skdprint("Checking functions")
            val iterator: HatIterator = new HatIterator(num, num)
            var assignment: List[Int] = iterator.next
            while (assignment != Nil) {
                if (!checkIfOneCorrect(assignment, guessingFunction)) {
                    skdprint(assignment.toString)
                    synthAssert(false)
                }
                assignment = iterator.next
            }
        }

        // checks that at least one player is correct for the assignment
        def checkIfOneCorrect(assignment: List[Int], guessingFunction: List[(List[Int] => Int)]): Boolean = {
            var people: List[(List[Int] => Int)] = guessingFunction
            var hats: List[Int] = assignment
            var seenHat: Boolean = false

            for (i <- 0 until num) {
                val person: List[Int] => Int = people.head
                people = people.tail
                val hat: Int = hats.head
                hats = hats.tail

                val (fronthalf, backhalf) = assignment.splitAt(i)
                var seenHats: List[Int] = fronthalf ::: backhalf.tail

                if (hat == person(seenHats)) {
                    seenHat = true
                }
            }

            return seenHat
        }

        val functions: List[(List[Int] => Int)] = getConstantGuessingFunction
        checkGuesses(functions)

        // print out the player tables
        val iterator: HatIterator = new HatIterator(num - 1, num)
        var value: List[Int] = iterator.next
        while (value != Nil) {
            var i: Int = 2;
            print(value + " | ");
            for (function <- functions) {
                val name = "" + i;
                val key = (value, name)
                if (colorMap.contains(key)) {
                    print(function(value).toString, colorMap(key))
                } else {
                    print(function(value).toString)
                }
                print(" | ")
                i -= 1
            }
            println("");
            value = iterator.next
        }
    }
}

object HatGame1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        //val cmdopts = new cli.CliParser(args)
        skalch.AngelicSketchSynthesize(() =>
            new HatGame1Sketch())
    }
}
