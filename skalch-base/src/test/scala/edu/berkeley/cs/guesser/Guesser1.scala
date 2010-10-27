
package edu.berkeley.cs.guesser

import sketch.util._
import skalch.AngelicSketch
import scala.collection.immutable.List

class Guesser1Sketch extends AngelicSketch {
    val tests = Array( () )
    
    
    class Guesser(val d1 : Int, val d2 : Int, val d3 : Int, val d4 : Int) {
        
        var actualAnswer : Tuple4[Int, Int, Int, Int] = (d1, d2, d3, d4);
        var possibleAnswers : List[Tuple4[Int, Int, Int, Int]] = Nil;
        createAllAnswers()
        
        def createAllAnswers() {
            possibleAnswers = for (i1 <- List.range(1, 6);
                 i2 <- List.range(1, 7);
                 i3 <- List.range(1, 7);
                 i4 <- List.range(1, 7))
                yield (i1, i2, i3, i4)
        }
        
        def guessAnswer(guess : Tuple4[Int, Int, Int, Int], 
                answer : Tuple4[Int, Int, Int, Int]) : Tuple2[Int, Int] = {
            var posCorrect : Int = 0
            var valCorrect : Int = 0

            val answerBin = new Array[Int](7)
            val guessBin = new Array[Int](7)
            
            for (i <- 0 to 3) {
                val answerVal : Int = answer.productElement(i).asInstanceOf[Int]
                val guessVal : Int = guess.productElement(i).asInstanceOf[Int]
                
                if (guessVal == answerVal) {
                    posCorrect += 1
                } else {
                    answerBin(answerVal) += 1
                    guessBin(guessVal) += 1
                }
            }
            
            for (i <- 1 to 6) {
                valCorrect += Math.min(answerBin(i), guessBin(i));
            }
            
            return (posCorrect, valCorrect)
        }
        
        def filter(guess : Tuple4[Int, Int, Int, Int], response : Tuple2[Int, Int]) {
            possibleAnswers = possibleAnswers.filter( value => 
                    guessAnswer(guess, value).equals(response))
        }
        
        def submitGuess(guess : Tuple4[Int, Int, Int, Int]) {
            val response : Tuple2[Int , Int] = guessAnswer(guess, actualAnswer)
            skdprint("Response: " + response)
            filter(guess, response)
        }
        
        def getNumPossibleAnswers() : Int = {
            return possibleAnswers.length
        }
        
    }
    
    def main() {
        val guesser = new Guesser(1,2,3,6);
        var prevNumAnswers = guesser.getNumPossibleAnswers();
        var finished : Boolean = false;
        
        skdprint("Answers remaining: " + prevNumAnswers)
              
        for (i <- 1 to 8) {
            if (!finished) {
                val i1 : Int = !!(List(1,2,3,4,5,6));
                val i2 : Int = !!(List(1,2,3,4,5,6));
                val i3 : Int = !!(List(1,2,3,4,5,6));
                val i4 : Int = !!(List(1,2,3,4,5,6));
                
                val guess : Tuple4[Int, Int, Int, Int] = (i1, i2, i3, i4)
                skdprint("Guess: " + guess)
                
                guesser.submitGuess(guess)
                
                val numAnswers : Int = guesser.getNumPossibleAnswers()
                skdprint("Answers remaining: " + numAnswers)
                
                if (numAnswers <= 1) {
                    finished = true;
                } else if (numAnswers > 5) {
                    synthAssert(numAnswers < (prevNumAnswers * 4) / 5);
                } else {
                    synthAssert(numAnswers < prevNumAnswers)
                }
                prevNumAnswers = numAnswers
            }
        }
        synthAssert(guesser.getNumPossibleAnswers() == 1);
    }
}

object Guesser1 {
    def main(args: Array[String]) = {
        for (arg <- args)
            Console.println(arg)
        skalch.AngelicSketchSynthesize(() => 
            new Guesser1Sketch())
        }
    }
