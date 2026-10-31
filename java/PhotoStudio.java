import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import javax.swing.border.EmptyBorder;

/**
 * Tweede Java-stap: studio metadata + gallery sidecar naast PhotorealViewer.
 */
public class PhotoStudio {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java PhotoStudio <image.bmp|png> [title]");
            System.exit(1);
        }
        Path imgPath = Path.of(args[0]).toAbsolutePath();
        String title = args.length > 1 ? args[1] : "Ultra Photoreal";
        BufferedImage img = ImageIO.read(imgPath.toFile());
        if (img == null) throw new IllegalArgumentException("Kan beeld niet lezen: " + imgPath);

        Path side = imgPath.resolveSibling(stripExt(imgPath.getFileName().toString()) + "_studio.txt");
        String meta = ""
                + "NeoTorch PhotoStudio (Java)\n"
                + "============================\n"
                + "title:    " + title + "\n"
                + "file:     " + imgPath.getFileName() + "\n"
                + "size:     " + img.getWidth() + " x " + img.getHeight() + "\n"
                + "pipeline: JS · Java · Python · Java · C++ · Rust · C\n"
                + "mode:     ULTRA PHOTOREAL\n"
                + "time:     " + LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + "\n";
        Files.writeString(side, meta);

        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("NeoTorch PhotoStudio · " + title);
            f.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            JPanel root = new JPanel(new BorderLayout(8, 8));
            root.setBackground(new Color(12, 14, 18));
            root.setBorder(new EmptyBorder(12, 12, 12, 12));

            JLabel head = new JLabel("PHOTO STUDIO  ·  Java metadata pass");
            head.setForeground(new Color(232, 168, 74));
            head.setFont(new Font("Segoe UI", Font.BOLD, 14));
            root.add(head, BorderLayout.NORTH);

            JTextArea area = new JTextArea(meta);
            area.setEditable(false);
            area.setFont(new Font("Consolas", Font.PLAIN, 13));
            area.setBackground(new Color(22, 26, 34));
            area.setForeground(new Color(220, 226, 236));
            area.setBorder(new EmptyBorder(10, 10, 10, 10));
            root.add(new JScrollPane(area), BorderLayout.CENTER);

            JLabel foot = new JLabel("Sidecar: " + side.getFileName());
            foot.setForeground(new Color(140, 150, 168));
            foot.setFont(new Font("Segoe UI", Font.PLAIN, 11));
            root.add(foot, BorderLayout.SOUTH);

            f.setContentPane(root);
            f.setSize(520, 320);
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });

        System.out.println("[Java/PhotoStudio] Wrote " + side);
    }

    private static String stripExt(String name) {
        int i = name.lastIndexOf('.');
        return i > 0 ? name.substring(0, i) : name;
    }
}
