package cuda
import skalch._
import skalch.cuda._
import skalch.cuda.annotations._
import sketch.util._

/**
 * accumulate things in the histogram
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required,
 *          if you make changes, please consider contributing back!
 */
class Histogram() extends CudaKernel {
    type Word = Int
    type IntArr = ScIArray1D[IntPtr]
    type WordArr = ScIArray1D[Word]

    class Sentence {
        val en : WordArr @ ArrayLen(40) @ scIField = null
        val fr : WordArr @ ArrayLen(40) @ scIField = null
    }

    class BloomFilter {
        val keys : IntArr = null

        def increment(idx : Int, amnt : Int) {
            val idx2 = if (idx < 0) -idx else idx
            keys(idx2 % keys.length).atomicAdd(amnt)
        }
    }

    /*

    struct VLArray_Int {
        int length;
        int *values;
    }

    struct BloomFilter {
        VLArray_Int keys;
    }

    __device__ void increment(int idx, int amnt) {
        int idx2;
        if (idx < 0) {
            idx2 = -idx;
        } else {
            idx2 = idx;
        }
        atomicAdd(&keys.values[idx2 % keys.length], amnt);
    }

    */

    def hash1(en : Word, fr : Word) = en * 7130881 + fr * 240727
    def hash2(en : Word, fr : Word) = en * 207139 + fr * 143797

    @scKernel def addToHistogram(
            sentences : ScIArray1D[Sentence],
            bigram : BloomFilter @ scCopy)
    {
        var sent_idx = blockIdx.x
        while (sent_idx < sentences.length) {
            val sent = sentences(sent_idx)

            var en_idx = threadIdx.x
            while (en_idx < sent.en.length) {
                val en_word = sent.en(en_idx)
                if (en_word != -1) {
                    // no "break" stmt

                    var fr_idx = threadIdx.y
                    while (fr_idx < sent.fr.length) {
                        val fr_word = sent.fr(fr_idx)
                        if (fr_word != -1) {
                            // no "break" stmt

                            bigram.increment(hash1(en_word, fr_word), 1)
                            bigram.increment(hash2(en_word, fr_word), 1)
                        }

                        fr_idx += blockDim.y
                    }
                }

                en_idx += blockDim.x
            }

            sent_idx += gridDim.x
            __syncthreads()
        }
    }
}
