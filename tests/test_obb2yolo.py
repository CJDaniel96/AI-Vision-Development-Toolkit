import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from obb2yolo import OBBSample, load_class_names, main, normalize_split, prepare_dataset, split_samples
from yolo_obb_utils import load_yolo_obb_annotations, save_yolo_obb_annotations


BOXES = [
    [[-0.05, 0.5], [0.5, -0.875], [1.05, 0.5], [0.5, 1.875]],
    [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5], [0.1, 0.5]],
]


class ClassNameTests(unittest.TestCase):
    def test_txt_and_yaml_preserve_id_order_and_unicode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, text in (
                ('classes.txt', '\ufeff螺絲\n\n連接器\n'),
                ('data.yaml', 'names: [螺絲, 連接器]\n'),
                ('mapping.yml', 'names: {1: 連接器, 0: 螺絲}\n'),
            ):
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_text(text, encoding='utf-8')
                    self.assertEqual(load_class_names(path), ['螺絲', '連接器'])

    def test_invalid_name_mappings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data.yaml'
            for text in (
                'names: {1: screw}', 'names: {0: screw, 2: pin}',
                'names: [screw, screw]', 'names: []', 'names: [null]',
                'names: {0.5: screw}', 'names: {true: screw}', 'nc: 2',
            ):
                with self.subTest(text=text):
                    path.write_text(text, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_class_names(path)


class SplitTests(unittest.TestCase):
    def make_samples(self, class_labels):
        return [
            OBBSample(Path(f'{index}.png'), Path(f'{index}.png'), [], labels)
            for index, labels in enumerate(class_labels)
        ]

    def test_overall_counts_and_common_class_combinations_are_balanced(self):
        samples = self.make_samples([[0, 1]] * 10 + [[2]] * 10)
        groups = split_samples(samples, (0.7, 0.2, 0.1))
        self.assertEqual({name: len(group) for name, group in groups.items()}, {
            'train': 14, 'val': 4, 'test': 2,
        })
        for name, count in (('train', 7), ('val', 2), ('test', 1)):
            self.assertEqual(sum(sample.class_labels == [0, 1] for sample in groups[name]), count)
        paths = [sample.relative_path for group in groups.values() for sample in group]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), {sample.relative_path for sample in samples})

    def test_rare_groups_and_small_datasets_keep_train_and_val_nonempty(self):
        for count in (2, 3, 6):
            for ratios in ((0.7, 0.2, 0.1), (0.01, 0.01, 0.98)):
                with self.subTest(count=count, ratios=ratios):
                    groups = split_samples(self.make_samples([[index] for index in range(count)]), ratios)
                    self.assertGreater(len(groups['train']), 0)
                    self.assertGreater(len(groups['val']), 0)
                    self.assertEqual(sum(map(len, groups.values())), count)
                    self.assertEqual(bool(groups['test']), count >= 3)

    def test_seed_reproduces_split_and_two_weights_disable_test(self):
        samples = self.make_samples([[0]] * 20)
        first = split_samples(samples, (8, 2), seed=11)
        self.assertEqual(first, split_samples(list(reversed(samples)), (8, 2), seed=11))
        self.assertNotEqual(first, split_samples(samples, (8, 2), seed=12))
        self.assertEqual({name: len(group) for name, group in first.items()}, {
            'train': 16, 'val': 4, 'test': 0,
        })

    def test_backgrounds_do_not_displace_all_labeled_images_from_train(self):
        samples = self.make_samples([[]] * 9 + [[0]])
        for ratios in ((0.7, 0.2, 0.1), (0.01, 0.01, 0.98)):
            with self.subTest(ratios=ratios):
                groups = split_samples(samples, ratios)
                self.assertTrue(any(sample.class_labels for sample in groups['train']))
                self.assertEqual(sum(map(len, groups.values())), 10)

    def test_invalid_ratios_and_single_image_are_rejected(self):
        for ratios in ((1,), (1, 2, 3, 4), (0, 1), (1, 0), (0.7, 0.4, -0.1), (1, float('nan'))):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                normalize_split(ratios)
        with self.assertRaisesRegex(ValueError, '至少需要 2 張'):
            split_samples(self.make_samples([[0]]), (0.8, 0.2))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.output = self.root / 'output'

    def write_sample(self, name, boxes=None, labels=None, images_dir=None, labels_dir=None):
        images_dir = images_dir or self.source
        labels_dir = labels_dir or images_dir
        image_path = images_dir / f'{name}.png'
        label_path = labels_dir / f'{name}.txt'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.random.default_rng(0).integers(0, 256, (80, 200, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        save_yolo_obb_annotations(label_path, BOXES[1:] if boxes is None else boxes, [0] if labels is None else labels)
        return image_path, label_path

    def run_dataset(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return prepare_dataset(self.source, self.output, **kwargs)

    def test_padding_and_yaml_are_ready_for_obb_training(self):
        image_path, label_path = self.write_sample('outside', BOXES, [0, 2])
        original_image, original_label = image_path.read_bytes(), label_path.read_bytes()
        self.write_sample('normal')
        self.write_sample('background', [], [])
        (self.source / 'classes.txt').write_text('螺絲\n連接器\n缺陷\n', encoding='utf-8')
        summary = self.run_dataset()
        self.assertEqual(summary.split_counts, {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual((summary.padded_images, summary.skipped_sources), (1, 0))
        config = yaml.safe_load((self.output / 'data.yaml').read_text())
        self.assertEqual(config['path'], str(self.output.resolve()))
        self.assertEqual(config['names'], ['螺絲', '連接器', '缺陷'])
        self.assertEqual(config['nc'], 3)
        for split in ('train', 'val', 'test'):
            self.assertTrue((self.output / config[split]).is_dir())
        images = list((self.output / 'images').rglob('*.png'))
        self.assertEqual(len(images), 3)
        for path in images:
            output_label = self.output / 'labels' / path.relative_to(self.output / 'images').with_suffix('.txt')
            boxes, labels = load_yolo_obb_annotations(output_label)
            self.assertTrue(np.all(np.asarray(boxes) >= 0))
            self.assertTrue(np.all(np.asarray(boxes) <= 1))
            if path.stem == 'outside':
                corrected = cv2.imread(str(path))
                self.assertEqual(corrected.shape, (220, 220, 3))
                np.testing.assert_array_equal(corrected[70:150, 10:210], cv2.imread(str(image_path)))
                np.testing.assert_allclose(np.asarray(boxes) * [220, 220], np.asarray(BOXES) * [200, 80] + [10, 70], atol=2e-6)
                self.assertEqual(labels, [0, 2])
            elif path.stem == 'normal':
                self.assertEqual(path.read_bytes(), (self.source / 'normal.png').read_bytes())
            else:
                self.assertEqual(output_label.read_text(), '')
        self.assertEqual(image_path.read_bytes(), original_image)
        self.assertEqual(label_path.read_bytes(), original_label)

    def test_cvat_layout_preserves_relative_paths_and_yaml_class_ids(self):
        for folder in ('batch1', 'batch2'):
            self.write_sample(f'{folder}/same', labels=[1], images_dir=self.source / 'images', labels_dir=self.source / 'labels')
        (self.source / 'data.yaml').write_text('names: {1: pin, 0: screw}\n', encoding='utf-8')
        summary = self.run_dataset(split=(0.8, 0.2))
        self.assertEqual(summary.names, ['screw', 'pin'])
        self.assertEqual(summary.split_counts, {'train': 1, 'val': 1, 'test': 0})
        config = yaml.safe_load((self.output / 'data.yaml').read_text())
        self.assertNotIn('test', config)
        paths = list((self.output / 'labels').rglob('same.txt'))
        self.assertEqual({path.parent.name for path in paths}, {'batch1', 'batch2'})
        self.assertTrue(all(load_yolo_obb_annotations(path)[1] == [1] for path in paths))

    def test_sparse_ids_get_placeholder_names_without_remapping(self):
        for name in ('one', 'two'):
            self.write_sample(name, labels=[3])
        summary = self.run_dataset()
        self.assertEqual(summary.names, ['class_0', 'class_1', 'class_2', 'class_3'])
        for path in (self.output / 'labels').rglob('*.txt'):
            self.assertEqual(load_yolo_obb_annotations(path)[1], [3])

    def test_missing_invalid_and_corrupt_sources_are_skipped(self):
        for name in ('one', 'two', 'missing', 'invalid', 'corrupt'):
            self.write_sample(name)
        (self.source / 'missing.txt').unlink()
        (self.source / 'invalid.txt').write_text('0 nan 0 1 0 1 1 0 1')
        (self.source / 'corrupt.png').write_bytes(b'not an image')
        summary = self.run_dataset()
        self.assertEqual(summary.skipped_sources, 3)
        self.assertEqual(sum(summary.split_counts.values()), 2)

    def test_missing_labels_can_be_explicitly_treated_as_background(self):
        self.write_sample('one')
        _, path = self.write_sample('background')
        path.unlink()
        summary = self.run_dataset(missing_as_background=True)
        self.assertEqual(summary.skipped_sources, 0)
        label_path = next((self.output / 'labels').rglob('background.txt'))
        self.assertEqual(label_path.read_text(), '')

    def test_wrong_class_list_fails_before_creating_output(self):
        self.write_sample('one', labels=[1])
        self.write_sample('two')
        path = self.root / 'classes.txt'
        path.write_text('only_class_zero\n')
        with self.assertRaisesRegex(ValueError, 'class_id 超出'), contextlib.redirect_stderr(io.StringIO()):
            self.run_dataset(classes_path=path)
        self.assertFalse(self.output.exists())

    def test_duplicate_stems_and_existing_output_are_rejected(self):
        image, _ = self.write_sample('one')
        self.write_sample('two')
        duplicate = self.source / 'one.jpg'
        duplicate.write_bytes(image.read_bytes())
        with self.assertRaisesRegex(ValueError, '重複影像檔名'):
            self.run_dataset()
        duplicate.unlink()
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, '輸出目錄已存在'):
            self.run_dataset()
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_write_failure_removes_staging_and_never_publishes_partial_dataset(self):
        self.write_sample('outside', BOXES, [0, 1])
        self.write_sample('two')
        with patch('obb2yolo.cv2.imwrite', return_value=False):
            with self.assertRaisesRegex(OSError, '寫入失敗'):
                self.run_dataset()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.output-*')), [])

    def test_cli_supports_explicit_separate_labels_and_class_yaml(self):
        for name in ('one', 'two'):
            self.write_sample(name, labels_dir=self.root / 'annotations')
        path = self.root / 'names.yaml'
        path.write_text('names: [screw]\n')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main([
                '--input', str(self.source), '--labels', str(self.root / 'annotations'),
                '--output', str(self.output), '--classes', str(path), '--split', '8', '2',
            ])
        self.assertEqual(code, 0)
        self.assertEqual(len(list((self.output / 'labels').rglob('*.txt'))), 2)

    def test_invalid_cli_and_unsafe_output_do_not_write(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['-i', str(self.source), '-o', str(self.output), '--split', '1', '0'])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, '輸出目錄不可'):
            prepare_dataset(self.source, self.root)


if __name__ == '__main__':
    unittest.main()
