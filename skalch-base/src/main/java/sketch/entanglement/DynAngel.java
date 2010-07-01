package sketch.entanglement;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DynAngel implements Comparable<DynAngel> {

    private static Pattern dynAngelPattern = Pattern.compile("^([\\d]+)\\[([\\d]+)\\]$");

    public final int staticAngelId;
    public final int execNum;

    public DynAngel(int staticAngelId, int execNum) {
        this.staticAngelId = staticAngelId;
        this.execNum = execNum;
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof DynAngel) {
            DynAngel otherLoc = (DynAngel) object;
            return staticAngelId == otherLoc.staticAngelId && execNum == otherLoc.execNum;
        }
        return false;
    }

    @Override
    public String toString() {
        return "" + staticAngelId + "[" + execNum + "]";
    }

    @Override
    public int hashCode() {
        return 1024 * staticAngelId + execNum;
    }

    public int compareTo(DynAngel otherAngel) {
        if (staticAngelId < otherAngel.staticAngelId) {
            return -1;
        } else if (staticAngelId > otherAngel.staticAngelId) {
            return 1;
        } else if (execNum < otherAngel.execNum) {
            return -1;
        } else if (execNum > otherAngel.execNum) {
            return 1;
        } else {
            return 0;
        }
    }

    public static DynAngel parseDynAngel(String input) {
        Matcher matcher = dynAngelPattern.matcher(input.trim());
        if (matcher.find()) {
            int staticId = Integer.parseInt(matcher.group(1));
            int execNum = Integer.parseInt(matcher.group(2));
            return new DynAngel(staticId, execNum);
        }
        return null;
    }

    public static List<DynAngel> parseDynAngelList(String input) {
        StringTokenizer tokens = new StringTokenizer(input, " ,\\t\\n");
        List<DynAngel> returnList = new ArrayList<DynAngel>();
        while (tokens.hasMoreTokens()) {
            String nextToken = tokens.nextToken();
            DynAngel angel = parseDynAngel(nextToken);
            if (angel != null) {
                returnList.add(angel);
            }
        }
        return returnList;
    }

    public static List<List<DynAngel>> parseDynAngelPartitioning(String input) {
        StringTokenizer tokens = new StringTokenizer(input, "{}");
        List<List<DynAngel>> returnList = new ArrayList<List<DynAngel>>();
        while (tokens.hasMoreTokens()) {
            String nextToken = tokens.nextToken();
            List<DynAngel> angel = parseDynAngelList(nextToken);
            if (angel != null && !angel.isEmpty()) {
                returnList.add(angel);
            }
        }
        return returnList;
    }
}
