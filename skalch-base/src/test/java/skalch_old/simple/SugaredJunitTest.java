package skalch_old.simple;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import sketch.util.DebugOut;

public class SugaredJunitTest {
    @Test
    public void SugaredTest() {
        DebugOut.print("sugared test...");
        ScalaMainRunner.run("skalch_old.simple.SugaredTest", "--ui_no_gui");
        assertTrue("test assert", false);
    }
}
