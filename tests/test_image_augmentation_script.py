import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from image_augmentation_script import (
    make_transforms_resolver,
    run_annotated_augmentation,
    score_source_against_deficits,
    select_balance_source,
)


def write_voc_sample(directory: Path, stem: str, labels: list[str]):
    """寫出一張測試影像與對應的 PASCAL VOC XML，每個 label 一個 bbox。"""
    width, height = 200, 200
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(directory / f'{stem}.jpg'), image)

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = f'{stem}.jpg'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    # bbox 集中在影像中央，避免擴增後被 min_visibility 濾掉造成測試不穩定
    for index, label in enumerate(labels):
        xmin = 60 + index * 10
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = label
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = '60'
        ET.SubElement(bndbox, 'xmax').text = str(xmin + 40)
        ET.SubElement(bndbox, 'ymax').text = '140'
    ET.ElementTree(annotation).write(str(directory / f'{stem}.xml'), encoding='utf-8')


def count_output_instances(labels_dir: Path) -> Counter:
    counts: Counter = Counter()
    for xml_path in labels_dir.glob('*.xml'):
        root = ET.parse(xml_path).getroot()
        counts.update(obj.find('name').text for obj in root.findall('object'))
    return counts


class ScoreSourceAgainstDeficitsTests(unittest.TestCase):
    def test_splits_gain_and_waste_by_remaining_deficit(self):
        gain, waste = score_source_against_deficits(
            Counter({'a': 3, 'b': 2}), {'a': 1, 'b': 5}
        )

        self.assertEqual((gain, waste), (3, 2))

    def test_satisfied_label_counts_as_pure_waste(self):
        gain, waste = score_source_against_deficits(
            Counter({'a': 1, 'b': 4}), {'a': 2, 'b': -3}
        )

        self.assertEqual((gain, waste), (1, 4))


class SelectBalanceSourceTests(unittest.TestCase):
    def test_prefers_source_that_fills_most_without_overshooting(self):
        sources = [
            Counter({'a': 1}),
            Counter({'a': 1, 'b': 5}),
        ]

        # b 還缺很多 → 選同時帶 b 的來源
        self.assertEqual(
            select_balance_source(sources, {'a': 1, 'b': 100}, [0, 0]), 1
        )
        # b 已達標 → 帶 b 的來源變成純浪費，改選只含 a 的來源
        self.assertEqual(
            select_balance_source(sources, {'a': 1, 'b': 0}, [0, 0]), 0
        )

    def test_spreads_across_equivalent_sources_by_usage(self):
        sources = [Counter({'a': 1}), Counter({'a': 1})]

        self.assertEqual(select_balance_source(sources, {'a': 10}, [0, 0]), 0)
        self.assertEqual(select_balance_source(sources, {'a': 10}, [1, 0]), 1)

    def test_returns_none_when_no_source_can_fill_deficit(self):
        sources = [Counter({'a': 1})]

        self.assertIsNone(select_balance_source(sources, {'a': 0, 'b': 5}, [0]))

    def test_skips_blocked_sources(self):
        sources = [Counter({'a': 5}), Counter({'a': 1})]

        self.assertEqual(
            select_balance_source(sources, {'a': 10}, [0, 0], blocked={0}), 1
        )


class MakeTransformsResolverTests(unittest.TestCase):
    def test_advances_angle_only_for_rotation(self):
        resolve = make_transforms_resolver(
            rotation_transforms=[], is_fixed_step=True, rotate_step=90
        )

        angles = []
        for _ in range(3):
            resolve('mixed')  # 非 rotation 不應推進角度
            angles.append(resolve('rotation')[0].limit)

        self.assertEqual(angles, [(0, 0), (90, 90), (180, 180)])

    def test_returns_none_for_unknown_type(self):
        resolve = make_transforms_resolver(None, False, None)

        self.assertIsNone(resolve('does_not_exist'))
        self.assertIsNone(resolve('rotation'))


class AugmentPerClassTests(unittest.TestCase):
    def run_balance(self, samples: dict[str, list[str]], per_class_count: int):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / 'dataset'
            output = root / 'output'
            dataset.mkdir()

            for stem, labels in samples.items():
                write_voc_sample(dataset, stem, labels)

            run_annotated_augmentation(
                images_dir=dataset,
                labels_dir=dataset,
                output_dir=output,
                augmentation_types=['mixed'],
                num_augmentations=1,
                rotate_limit=None,
                rotate_step=None,
                augment_per_class=per_class_count,
            )

            counts = count_output_instances(output / 'labels')
            image_count = len(list((output / 'images').glob('*.jpg')))
            return counts, image_count

    def test_every_label_reaches_the_target_instance_count(self):
        counts, image_count = self.run_balance(
            {
                'a_only': ['a'],
                'b_only': ['b'],
                'c_only': ['c'],
            },
            per_class_count=20,
        )

        self.assertEqual(dict(counts), {'a': 20, 'b': 20, 'c': 20})
        self.assertEqual(image_count, 60)

    def test_imbalanced_source_counts_still_reach_the_target(self):
        # a 有 3 張來源、b 只有 1 張，兩者都要補到 15
        samples = {f'a_{index}': ['a'] for index in range(3)}
        samples['b_0'] = ['b']

        counts, _ = self.run_balance(samples, per_class_count=15)

        self.assertEqual(dict(counts), {'a': 15, 'b': 15})

    def test_multi_label_image_overshoots_only_where_unavoidable(self):
        # 唯一含 b 的影像同時含 a，補 b 必然讓 a 超額
        counts, _ = self.run_balance(
            {
                'a_only': ['a'],
                'a_and_b': ['a', 'b'],
            },
            per_class_count=10,
        )

        self.assertEqual(counts['b'], 10)
        self.assertGreaterEqual(counts['a'], 10)

    def test_output_contains_no_original_files(self):
        counts, _ = self.run_balance({'a_only': ['a']}, per_class_count=5)

        self.assertEqual(dict(counts), {'a': 5})


if __name__ == '__main__':
    unittest.main()
