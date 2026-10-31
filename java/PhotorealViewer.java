import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;
import javax.swing.border.EmptyBorder;

/**
 * NeoTorch fotorealistische viewer (Java Swing).
 * Toont BMP/PNG van de C++ path tracer.
 */
public class PhotorealViewer extends JFrame {
    private final BufferedImage image;
    private final String titleText;

    public PhotorealViewer(BufferedImage image, String path) {
        this.image = image;
        this.titleText = path;
        setTitle("NeoTorch · Photoreal — " + Path.of(path).getFileName());
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        setBackground(new Color(12, 14, 18));

        JPanel root = new JPanel(new BorderLayout());
        root.setBackground(new Color(12, 14, 18));
        root.setBorder(new EmptyBorder(12, 12, 12, 12));

        JLabel header = new JLabel("FOTOREALISTISCH  ·  C++ path tracer  ·  Java viewer");
        header.setForeground(new Color(232, 168, 74));
        header.setFont(new Font("Segoe UI", Font.BOLD, 14));
        header.setBorder(new EmptyBorder(0, 4, 10, 4));
        root.add(header, BorderLayout.NORTH);

        JPanel canvas = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g;
                g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g2.setColor(new Color(18, 20, 26));
                g2.fillRect(0, 0, getWidth(), getHeight());
                if (image == null) return;
                int iw = image.getWidth();
                int ih = image.getHeight();
                double scale = Math.min(
                        (getWidth() - 16.0) / iw,
                        (getHeight() - 16.0) / ih);
                scale = Math.min(scale, 2.0);
                int dw = Math.max(1, (int) Math.round(iw * scale));
                int dh = Math.max(1, (int) Math.round(ih * scale));
                int x = (getWidth() - dw) / 2;
                int y = (getHeight() - dh) / 2;
                g2.drawImage(image, x, y, dw, dh, null);
            }

            @Override
            public Dimension getPreferredSize() {
                if (image == null) return new Dimension(640, 480);
                return new Dimension(image.getWidth() + 24, image.getHeight() + 24);
            }
        };
        canvas.setBackground(new Color(12, 14, 18));
        JScrollPane scroll = new JScrollPane(canvas);
        scroll.getViewport().setBackground(new Color(12, 14, 18));
        scroll.setBorder(null);
        root.add(scroll, BorderLayout.CENTER);

        JLabel foot = new JLabel(titleText, SwingConstants.LEFT);
        foot.setForeground(new Color(140, 150, 168));
        foot.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        foot.setBorder(new EmptyBorder(8, 4, 0, 4));
        root.add(foot, BorderLayout.SOUTH);

        setContentPane(root);
        setSize(Math.min(1100, Math.max(720, image.getWidth() + 80)),
                Math.min(900, Math.max(560, image.getHeight() + 120)));
        setLocationRelativeTo(null);
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java PhotorealViewer <image.bmp|png>");
            System.exit(1);
        }
        String path = args[0];
        SwingUtilities.invokeLater(() -> {
            try {
                BufferedImage img = ImageIO.read(new File(path));
                if (img == null) {
                    throw new IllegalArgumentException("Kan afbeelding niet lezen: " + path);
                }
                PhotorealViewer v = new PhotorealViewer(img, path);
                v.setVisible(true);
            } catch (Exception e) {
                e.printStackTrace();
                JFrame err = new JFrame("NeoTorch Photoreal");
                err.add(new JLabel("Fout: " + e.getMessage()));
                err.setSize(480, 120);
                err.setLocationRelativeTo(null);
                err.setVisible(true);
            }
        });
    }
}
