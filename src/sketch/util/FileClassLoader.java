package sketch.util;

import java.io.File;
import java.io.InputStream;
import java.util.regex.Pattern;

/**
 * This file is also at http://www.gatoatigrado.com/code-snippets; please keep
 * it in sync.
 * @author gatoatigrado (Nicholas Tung) [email ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class FileClassLoader extends ClassLoader {
    public File base_dir;
    public Pattern exclude_filter;

    public FileClassLoader(String base_dir, Pattern exclude_filter) {
        this.base_dir = new File(base_dir);
        this.exclude_filter = exclude_filter;
        if (!this.base_dir.isDirectory()) {
            System.err
                    .println("FileClassLoader -- existing directory expected, got "
                            + base_dir);
        }
    }

    @Override
    public Class<?> loadClass(String name) throws ClassNotFoundException {
        File cls_file = new File(base_dir.getAbsoluteFile() + File.separator
                + name.replaceAll("\\.", File.separator) + ".class");
        boolean exclude = (exclude_filter != null)
                && (exclude_filter.matcher(name).find());

        if (cls_file.isFile() && !exclude) {
            // load it from the bytecode file
            try {
                InputStream stream = cls_file.toURI().toURL().openStream();
                byte[] clsdef = new byte[stream.available()];
                if (stream.read(clsdef) == clsdef.length) {
                    // -ea isn't default, so don't rely on assertions
                    return defineClass(name, clsdef, 0, clsdef.length);
                }
            } catch (Exception e) {
                System.err.println("Failure reading class file "
                        + cls_file.getAbsolutePath());
                e.printStackTrace(); // still fall through
            }
        }

        return super.loadClass(name);
    }
}
