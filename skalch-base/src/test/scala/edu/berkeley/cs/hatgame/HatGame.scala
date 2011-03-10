
package edu.berkeley.cs.hatgame

import scala.collection.immutable.HashMap
import java.awt.Color

import skalch.AngelicSketch
import sketch.util._

class HatGameSketch extends AngelicSketch {
  val tests = Array(())

  def main() {
    val num: Int = 3

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

    class LimitedHatIterator(examples: List[List[Int]]) {
      var curExamples: List[List[Int]] = examples

      def next(): List[Int] = {
        if (curExamples != Nil) {
          val example: List[Int] = curExamples.head
          curExamples = curExamples.tail
          return example
        } else {
          return Nil
        }
      }
    }

    def getFunction(): (List[Int] => Int, List[Int] => Color) = {
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
      var colorMap: Map[List[Int], Color] = new HashMap[List[Int], Color]

      val function: List[Int] => Int = (input => { 
        if (!valueMap.contains(input)) {
          valueMap += input -> !!(output)
          val c = sklast_angel_color()
          colorMap += input -> c
          skdprint("Creating input: " + input + " -> " + valueMap(input), c)
        }
        valueMap(input)
      })
      
      val colorFunction: List[Int] => Color = (input => { 
        if (!colorMap.contains(input)) {
          colorMap(input)
        } else {
          Color.black
        }
      })
      
      return (function, colorFunction)
    }

    def getGuessingFunction(): List[(List[Int] => Int)] = {
      var guessingFunctions: List[(List[Int] => Int)] = Nil
      for (i <- 0 until num) {
        guessingFunctions ::= getFunction()._1
      }
      return guessingFunctions
    }

    def getConstantFunction(valueMap : Map[List[Int], Int]) : List[Int] => Int = {
        return (input => valueMap(input))
    }
    
    def getConstantGuessingFunction(): List[(List[Int] => Int)] = {
      var guessingFunctions: List[(List[Int] => Int)] = Nil
      
      var valueMap : Map[List[Int], Int] = new HashMap[List[Int], Int]()

//      fail
//      valueMap += List(0,0) -> 0
//      valueMap += List(0,1) -> 1
//      valueMap += List(0,2) -> 2
//      valueMap += List(1,0) -> 0
//      valueMap += List(1,1) -> 1
//      valueMap += List(1,2) -> 2
//      valueMap += List(2,0) -> 0
//      valueMap += List(2,1) -> 1
//      valueMap += List(2,2) -> 2

      valueMap += List(0,0) -> 1
      valueMap += List(0,1) -> 0
      valueMap += List(0,2) -> 0
      valueMap += List(1,0) -> 2
      valueMap += List(1,1) -> 2
      valueMap += List(1,2) -> 0
      valueMap += List(2,0) -> 2
      valueMap += List(2,1) -> 1
      valueMap += List(2,2) -> 1
      
//      valueMap += List(0,0) -> 0
//      valueMap += List(0,1) -> 1
//      valueMap += List(0,2) -> 2
//      valueMap += List(1,0) -> 1
//      valueMap += List(1,1) -> 2
//      valueMap += List(1,2) -> 0
//      valueMap += List(2,0) -> 2
//      valueMap += List(2,1) -> 0
//      valueMap += List(2,2) -> 1

      
      guessingFunctions ::= getConstantFunction(valueMap)
      for (i <- 0 until num-1) {
        guessingFunctions ::= getFunction()._1
      }
      return guessingFunctions
    }

    def checkGuesses(guessingFunction: List[(List[Int] => Int)]) {
      skdprint("Checking functions")
      val iterator: HatIterator = new HatIterator(num, num)
      var assignment: List[Int] = iterator.next
      while (assignment != Nil) {
        skdprint("** Hats: " + assignment.toString() + " **")
        if (!checkIfOneCorrect(assignment, guessingFunction)) {
          skdprint(assignment.toString)
          synthAssert(false)
        }
        assignment = iterator.next
      }
    }

    def checkGuessesLimited(guessingFunction: List[(List[Int] => Int)], tests : List[List[Int]]) {
      skdprint("Checking functions")
      val iterator: LimitedHatIterator = new LimitedHatIterator(tests)
      var assignment: List[Int] = iterator.next
      while (assignment != Nil) {
        if (!checkIfOneCorrect(assignment, guessingFunction)) {
          skdprint(assignment.toString)
          synthAssert(false)
        }
        assignment = iterator.next
      }
    }
    def checkIfOneCorrect(assignment: List[Int], guessingFunction: List[(List[Int] => Int)]): Boolean = {
      var people: List[(List[Int] => Int)] = guessingFunction
      var hats: List[Int] = assignment
      var seenHat : Boolean = false

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
    for (function <- functions) {
      skdprint("function")
      val iterator: HatIterator = new HatIterator(num - 1, num)
      var value: List[Int] = iterator.next
      while (value != Nil) {
        skdprint(value + "-->" + function(value))
        value = iterator.next
      }
    }
  }
}

object HatGame {
  def main(args: Array[String]) = {
    for (arg <- args)
      Console.println(arg)
    //val cmdopts = new cli.CliParser(args)
    skalch.AngelicSketchSynthesize(() =>
      new HatGameSketch())
  }
}
