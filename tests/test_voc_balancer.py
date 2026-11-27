import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from vision_toolkit.voc_balancer import augment_image, augment_xml, scan_dataset


VOC_SAMPLE = '''<annotation>
  <filename>sample.png</filename>
  <size><width>10</width><height>8</height></size>
  <object><name>part</name><bndbox>
    <xmin>2</xmin><ymin>1</ymin><xmax>5</xmax><ymax>3</ymax>
  </bndbox></object>
</annotation>'''


class VocBalancerTests(unittest.TestCase):
    def test_label_indexes_reference_retained_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a_empty.xml').write_text('<annotation><object><name/></object></annotation>')
            (root / 'b_part.xml').write_text(VOC_SAMPLE)
            samples, counts, indexes = scan_dataset(root, None)
            self.assertEqual(counts, {'part': 1})
            self.assertEqual(indexes['part'], [0])
            self.assertEqual(samples[0]['xml'].name, 'b_part.xml')

    def test_rotated_voc_box_matches_the_rotated_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / 'sample.xml'
            xml.write_text(VOC_SAMPLE)
            image_path = root / 'sample.png'
            image = Image.new('L', (10, 8))
            image.paste(255, (2, 1, 5, 3))
            image.save(image_path)

            for method in ('rotate90', 'rotate180'):
                with self.subTest(method=method):
                    out_xml, out_image = root / f'{method}.xml', root / f'{method}.png'
                    augment_xml(xml, out_xml, method, method)
                    augment_image(image_path, out_image, method)
                    annotation = ET.parse(out_xml).getroot()
                    box = annotation.find('./object/bndbox')
                    coordinates = tuple(float(box.findtext(key)) for key in ('xmin', 'ymin', 'xmax', 'ymax'))
                    width, height = (int(annotation.findtext(f'./size/{key}')) for key in ('width', 'height'))
                    with Image.open(out_image) as rotated:
                        self.assertEqual(rotated.getbbox(), coordinates)
                        self.assertEqual(rotated.size, (width, height))


if __name__ == '__main__':
    unittest.main()
