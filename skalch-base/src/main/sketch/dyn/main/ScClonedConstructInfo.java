package sketch.dyn.main;

/**
 * actual clone of construct info
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class ScClonedConstructInfo extends ScConstructInfo {
    public int uid_;
    public int untilv_;

    public ScClonedConstructInfo(ScConstructInfo prev) {
        uid_ = prev.uid();
        untilv_ = prev.untilv();
    }

    @Override
    public int uid() {
        return uid_;
    }

    @Override
    public int untilv() {
        return untilv_;
    }

    public static ScClonedConstructInfo[] clone_array(ScConstructInfo[] prev) {
        ScClonedConstructInfo[] result = new ScClonedConstructInfo[prev.length];
        for (int a = 0; a < prev.length; a++) {
            result[a] = new ScClonedConstructInfo(prev[a]);
        }
        return result;
    }
}
